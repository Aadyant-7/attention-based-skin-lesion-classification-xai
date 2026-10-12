"""One fixed, probability-only residual reranker for caller-owned meta folds.

This module reads no files and writes no artifacts. Cross-fitting protects only
this new fit; it does not make previously selected base CNNs independent of the
development cohort. The caller owns lesion-group splitting and all reporting.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


PROBABILITY_FLOOR = 1e-12
MARGIN_LIMIT = 0.5
L2_PENALTY = 0.01
MAX_ITERATIONS = 500
FEATURE_NAMES = (
    [f"leader_class_{i}" for i in range(7)]
    + [f"runner_class_{i}" for i in range(7)]
    + [f"member_{i}_runner_minus_leader" for i in range(6)]
)


def _checked_indices(values, size):
    indices = np.asarray(values)
    assert indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
    indices = indices.astype(np.int64, copy=False)
    assert len(indices) and len(np.unique(indices)) == len(indices)
    assert (indices >= 0).all() and (indices < size).all()
    return indices


def crossfit_top2(p, stack, y, folds):
    """Return held-out probabilities and fully reproducible fold parameters.

    ``p`` is N x 7 frozen S83 probabilities, ``stack`` is 6 x N x 7 member
    probabilities, ``y`` is N integer labels, and ``folds`` yields disjoint
    (training_indices, holdout_indices) pairs covering every row once. The caller
    must additionally establish lesion disjointness; lesion IDs are not features.

    Only eligible training rows fit the binary correction. Its positive class
    means the frozen runner-up matches the training label; all other labels are
    negative. Eligibility, features, and the baseline offset use predictions only.
    """
    p = np.asarray(p, dtype=np.float64)
    stack = np.asarray(stack, dtype=np.float64)
    y = np.asarray(y)
    assert p.ndim == 2 and p.shape[1] == 7
    size = len(p)
    assert size > 0 and stack.shape == (6, size, 7) and y.shape == (size,)
    assert np.issubdtype(y.dtype, np.integer) and ((y >= 0) & (y < 7)).all()
    assert np.isfinite(p).all() and np.isfinite(stack).all()
    assert (p >= 0).all() and (stack >= 0).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5, rtol=0)
    assert np.allclose(stack.sum(axis=2), 1.0, atol=1e-5, rtol=0)

    rows = np.arange(size)
    leader = p.argmax(axis=1)
    remaining = p.copy()
    remaining[rows, leader] = -1.0
    runner = remaining.argmax(axis=1)
    margin = p[rows, leader] - p[rows, runner]
    votes = stack.argmax(axis=2)
    disagreement = (votes != votes[0:1]).any(axis=0)
    eligible = (margin <= MARGIN_LIMIT) & disagreement

    class_features = np.eye(7, dtype=np.float64)
    member_differences = (
        stack[:, rows, runner] - stack[:, rows, leader]
    ).T
    features = np.concatenate(
        [class_features[leader], class_features[runner], member_differences],
        axis=1,
    )
    assert features.shape == (size, 20) and np.isfinite(features).all()
    offset = np.log(np.maximum(p[rows, runner], PROBABILITY_FLOOR)) - np.log(
        np.maximum(p[rows, leader], PROBABILITY_FLOOR)
    )
    assert np.isfinite(offset).all()

    output = p.copy()
    coverage = np.zeros(size, dtype=np.int64)
    records = []
    for fold_number, (train_values, hold_values) in enumerate(folds):
        train = _checked_indices(train_values, size)
        hold = _checked_indices(hold_values, size)
        assert not np.intersect1d(train, hold).size
        assert len(train) + len(hold) == size
        assert np.array_equal(np.sort(np.concatenate([train, hold])), rows)
        assert (coverage[hold] == 0).all()
        coverage[hold] += 1

        fit_rows = train[eligible[train]]
        assert len(fit_rows)
        x = features[fit_rows]
        target = (y[fit_rows] == runner[fit_rows]).astype(np.float64)
        assert (target == 1).any() and (target == 0).any()
        fit_offset = offset[fit_rows]

        def objective(parameters):
            coefficients, intercept = parameters[:-1], parameters[-1]
            logits = fit_offset + x @ coefficients + intercept
            losses = np.logaddexp(0.0, logits) - target * logits
            value = losses.mean() + 0.5 * L2_PENALTY * np.dot(
                parameters, parameters
            )
            errors = (expit(logits) - target) / len(target)
            gradient = np.concatenate([x.T @ errors, [errors.sum()]])
            gradient += L2_PENALTY * parameters
            assert np.isfinite(value) and np.isfinite(gradient).all()
            return float(value), gradient

        initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
        fitted = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "gtol": 1e-7},
        )
        assert fitted.success, str(fitted.message)
        assert np.isfinite(fitted.x).all()

        hold_eligible = hold[eligible[hold]]
        residual = features[hold_eligible] @ fitted.x[:-1] + fitted.x[-1]
        runner_probability = expit(offset[hold_eligible] + residual)
        pair_mass = (
            p[hold_eligible, leader[hold_eligible]]
            + p[hold_eligible, runner[hold_eligible]]
        )
        output[hold_eligible, leader[hold_eligible]] = pair_mass * (
            1.0 - runner_probability
        )
        output[hold_eligible, runner[hold_eligible]] = (
            pair_mass * runner_probability
        )

        records.append(
            dict(
                fold=int(fold_number),
                train_rows=train.tolist(),
                holdout_rows=hold.tolist(),
                fit_rows=fit_rows.tolist(),
                eligible_holdout_rows=hold_eligible.tolist(),
                fit_positive_runner_labels=int(target.sum()),
                fit_negative_labels=int(len(target) - target.sum()),
                feature_names=list(FEATURE_NAMES),
                coefficients=fitted.x[:-1].tolist(),
                intercept=float(fitted.x[-1]),
                l2_penalty=L2_PENALTY,
                objective=float(fitted.fun),
                optimizer_iterations=int(fitted.nit),
                optimizer_success=bool(fitted.success),
                optimizer_message=str(fitted.message),
                optimizer_max_iterations=MAX_ITERATIONS,
                optimizer_ftol=1e-12,
                optimizer_gtol=1e-7,
                zero_initialization=True,
                margin_limit=MARGIN_LIMIT,
                probability_floor=PROBABILITY_FLOOR,
                eligibility_uses_labels=False,
            )
        )

    assert len(records) == 5 and (coverage == 1).all()
    assert np.array_equal(output[~eligible], p[~eligible])
    assert np.isfinite(output).all() and (output >= 0).all()
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5, rtol=0)
    return output, records
