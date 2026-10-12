"""Fixed, regularized clinical residuals for supplied development folds.

This module performs no file, checkpoint, image or GPU access. The caller must
provide safely parsed development metadata and lesion-disjoint fold indices,
and must freeze the evaluation plan before invoking the fitting function.
Cross-fitting the residual does not remove prior CNN checkpoint selection on
the supplied development cohort.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


L2 = 0.01
EPSILON = 1e-12
OPTIMIZER_OPTIONS = dict(maxiter=500, ftol=1e-12, gtol=1e-7)
CLINICAL_COLUMNS = ('age', 'sex', 'localization')
UNKNOWN = frozenset(('', 'unknown', 'nan', 'none', 'null', 'n/a', 'na'))


def _category(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    return None if value in UNKNOWN else value


def _schema(metadata, train):
    """Derive all numeric statistics and vocabularies from fit rows only."""
    age = pd.to_numeric(metadata.iloc[train]['age'], errors='coerce').to_numpy(
        dtype=np.float64, na_value=np.nan)
    finite = age[np.isfinite(age)]
    if finite.size and ((finite < 0).any() or (finite > 120).any()):
        raise ValueError('Observed training age must lie between 0 and 120')
    mean = float(finite.mean()) if finite.size else 0.0
    raw_std = float(finite.std(ddof=0)) if finite.size else 0.0
    std = raw_std if raw_std > EPSILON else 1.0
    vocabularies = {}
    names = ['age_z']
    for field in ('sex', 'localization'):
        observed = [_category(v) for v in metadata.iloc[train][field].tolist()]
        vocabularies[field] = sorted({v for v in observed if v is not None})
        names.extend(field + '=' + v for v in vocabularies[field])
    return dict(
        age=dict(mean=mean, std=std, raw_std=raw_std,
                 observed_train_count=int(finite.size),
                 missing_policy='Training mean; standardized contribution zero'),
        vocabularies=vocabularies, feature_names=names,
        categorical_policy='Training-only vocabularies; unknown/missing/unseen all zero',
        intercept_name='intercept',
        parameter_order=['intercept'] + names)


def _encode(metadata, indices, schema):
    """Transform supplied rows with a previously frozen training schema."""
    subset = metadata.iloc[indices]
    age = pd.to_numeric(subset['age'], errors='coerce').to_numpy(
        dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(age)
    if ((age[finite] < 0).any() or (age[finite] > 120).any()):
        raise ValueError('Observed assessment age must lie between 0 and 120')
    age = np.where(finite, age, schema['age']['mean'])
    columns = [(age - schema['age']['mean']) / schema['age']['std']]
    for field in ('sex', 'localization'):
        values = [_category(v) for v in subset[field].tolist()]
        for value in schema['vocabularies'][field]:
            columns.append(np.asarray([v == value for v in values], dtype=np.float64))
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise FloatingPointError('Clinical features must be finite')
    return features


def _fit(offset, y, features):
    """Fit zero-initialized offset multinomial CE with all parameters penalized."""
    n, classes = offset.shape
    design = np.column_stack([np.ones(n, dtype=np.float64), features])
    shape = (design.shape[1], classes)

    def objective(flat):
        coefficients = flat.reshape(shape)
        logits = offset + design @ coefficients
        log_normalizer = logsumexp(logits, axis=1)
        value = float(np.mean(log_normalizer - logits[np.arange(n), y])
                      + 0.5 * L2 * np.sum(coefficients ** 2))
        residual = np.exp(logits - log_normalizer[:, None])
        residual[np.arange(n), y] -= 1.0
        gradient = design.T @ residual / n + L2 * coefficients
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite clinical residual objective or gradient')
        return value, gradient.ravel()

    initial = np.zeros(int(np.prod(shape)), dtype=np.float64)
    result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                      options=dict(OPTIMIZER_OPTIONS))
    if not result.success:
        raise RuntimeError('Clinical residual optimization did not converge: '
                           + str(result.message))
    coefficients = result.x.reshape(shape)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError('Nonfinite fitted clinical residual coefficients')
    information = dict(
        method='L-BFGS-B', options=dict(OPTIMIZER_OPTIONS), l2=L2,
        objective='Mean unweighted cross-entropy + 0.01/2 * sum(all parameters squared)',
        initial_correction='All parameters zero', success=bool(result.success),
        status=int(result.status), message=str(result.message),
        iterations=int(result.nit), function_evaluations=int(result.nfev),
        gradient_evaluations=int(result.njev), objective_value=float(result.fun),
        maximum_absolute_gradient=float(np.max(np.abs(result.jac))),
        coefficient_shape=list(shape), coefficients=coefficients.tolist())
    return coefficients, information


def _predict(offset, features, coefficients):
    design = np.column_stack([np.ones(len(offset), dtype=np.float64), features])
    logits = offset + design @ coefficients
    probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    if (not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)):
        raise FloatingPointError('Residual probabilities must be finite and normalized')
    return probabilities


def _indices(values, n, kind):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu' or not len(values):
        raise ValueError(kind + ' indices must be a nonempty one-dimensional integer array')
    values = values.astype(np.int64, copy=False)
    if (values < 0).any() or (values >= n).any() or len(np.unique(values)) != len(values):
        raise ValueError(kind + ' indices must be unique and within the supplied rows')
    return values


def crossfit_clinical(p, metadata, y, folds):
    """Return matched control/clinical OOF probabilities in original row order.

    ``folds`` is an iterable of ``(train_indices, hold_indices)`` pairs. Each row
    must be held out exactly once. Lesion grouping is the caller's obligation:
    identifiers are deliberately not accepted as clinical predictive features.
    Parameters use shape ``(intercept plus features, class index)`` and each fold
    returns the complete training schema needed to reconstruct its predictions.
    """
    p = np.asarray(p, dtype=np.float64)
    raw_y = np.asarray(y)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError('p must be a nonempty row-by-class probability matrix')
    n, classes = p.shape
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1 + 1e-6).any()
            or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError('Input probabilities must be finite, nonnegative and normalized')
    if raw_y.shape != (n,) or raw_y.dtype.kind not in 'iu':
        raise ValueError('y must contain one integer class index per supplied row')
    y = raw_y.astype(np.int64, copy=False)
    if (y < 0).any() or (y >= classes).any():
        raise ValueError('Class indices must match probability columns')
    if not isinstance(metadata, pd.DataFrame) or len(metadata) != n:
        raise ValueError('metadata must be a DataFrame aligned to the probability rows')
    if any(field not in metadata.columns for field in CLINICAL_COLUMNS):
        raise ValueError('metadata must contain age, sex and localization')
    metadata = metadata.loc[:, list(CLINICAL_COLUMNS)]

    validated_folds = []
    held = np.zeros(n, dtype=np.int64)
    for train, hold in folds:
        train = _indices(train, n, 'Training')
        hold = _indices(hold, n, 'Held-out')
        if np.intersect1d(train, hold).size:
            raise ValueError('Training and held-out indices must be disjoint')
        if not np.array_equal(np.unique(y[train]), np.arange(classes)):
            raise ValueError('Every class must be present in each training fold')
        held[hold] += 1
        validated_folds.append((train, hold))
    if not validated_folds or not np.all(held == 1):
        raise ValueError('Each supplied row must be held out exactly once')

    offset = np.log(np.maximum(p, EPSILON))
    control_oof = np.full_like(p, np.nan)
    clinical_oof = np.full_like(p, np.nan)
    records = []
    for fold, (train, hold) in enumerate(validated_folds):
        schema = _schema(metadata, train)
        train_features = _encode(metadata, train, schema)
        control, control_info = _fit(offset[train], y[train],
                                     np.empty((len(train), 0), dtype=np.float64))
        clinical, clinical_info = _fit(offset[train], y[train], train_features)
        # Assessment features are transformed only after fitting with train rows.
        hold_features = _encode(metadata, hold, schema)
        control_oof[hold] = _predict(offset[hold],
                                     np.empty((len(hold), 0), dtype=np.float64), control)
        clinical_oof[hold] = _predict(offset[hold], hold_features, clinical)
        records.append(dict(
            fold=fold, train_indices=train.tolist(), hold_indices=hold.tolist(),
            training_class_counts=np.bincount(y[train], minlength=classes).tolist(),
            class_column_indices=list(range(classes)), probability_floor=EPSILON,
            schema=schema, control=control_info, clinical=clinical_info))
    if not np.isfinite(control_oof).all() or not np.isfinite(clinical_oof).all():
        raise FloatingPointError('Cross-fitting must populate every OOF probability')
    return dict(control_oof=control_oof, clinical_oof=clinical_oof,
                fold_records=records)
