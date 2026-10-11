# Structured experiment packages

Updated 11 October 2026. This folder contains completed structured research, not an empty preparation area. Training is closed; opening these files launches nothing.

- S01 is the initial strict control; S02 onward include original exploratory backbone/attention/fusion comparisons.
- S28/S29/S30 are the separately trained strict final components. Their ensemble evidence is under [../final_strict/v1/](../final_strict/v1/).
- [convnext_base_original_v1/](convnext_base_original_v1/) contains the latest R203 original-split ConvNeXt-V2 Base package, preparation freeze and two bounded CPU fusions.
- Later short studies and the retained S83 ensemble live under [../short_screening/](../short_screening/).

Use [../master_experiment_registry.csv](../master_experiment_registry.csv) for exact run IDs, protocols, selectors and artifact paths. Packages save configuration/environment, history, metrics, predictions/probabilities, class scores and figures; training checkpoints remain local under `checkpoints/` at the recorded path. Preserve packages even when their folder naming differs.

For relevant comparisons, use the [reporting evidence index](../../reporting/evidence/INDEX.md). The old preparation text is archived with previous navigation.
