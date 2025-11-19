<!-- Copilot instructions tailored for ATMS-523-Module-6 -->
# .github/copilot-instructions.md

Short, actionable guidance to help AI coding agents be productive in this repository.

Project overview
- This repo is a collection of teaching Jupyter notebooks for ATMS 523 Module 6.
- Main analysis notebooks (examples to follow):
  - `M6N1_Feature_Engineering.ipynb` (feature transforms, TF-IDF, DictVectorizer)
  - `M6N2-Classification-titanic.ipynb` (Titanic classification pipeline)
  - `M6N3-Regression-house-prices.ipynb` (house prices regression pipeline)
  - `M6N4_plot_permutation_importance.ipynb` (permutation importance demo)
  - `M6N5_plot_partial_dependence.ipynb` (partial dependence / ICE plots)

Data layout and conventions
- Data files are under `data/` or in repository root: `titanic.csv`, `houseprice.csv`, `climate_indices.csv`, `1950-2021_actual_tornadoes.csv`.
- Notebooks read these CSVs with relative paths (e.g. `pd.read_csv('climate_indices.csv')`); do not change filenames or paths without updating notebooks.

Key code patterns (concrete, discoverable)
- Preprocessing uses `feature_engine` transformers (imports like `feature_engine.imputation as mdi`, `feature_engine.encoding as ce`) rather than custom helpers. Refer to `M6N2-Classification-titanic.ipynb` and `M6N3-Regression-house-prices.ipynb` for usage patterns.
- Pipelines: `sklearn.pipeline.Pipeline` / `make_pipeline` and `sklearn.compose.ColumnTransformer` are used consistently. Example: a numeric scaler + estimator wrapped in `Pipeline`.
- Model inspection: `sklearn.inspection.permutation_importance`, `PartialDependenceDisplay.from_estimator`, and (notebook mention) SHAP may be used — verify `SHAP` is installed before using.
- Plotting: `matplotlib.pyplot` for figures; some cells call `plt.savefig` and write images to `figures/`.

Dependencies (in-notebook imports)
- Expect at minimum: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `feature_engine`.
- Optional from notebooks: `shap` (mentioned in `ATMS 523 HW 6.md`), `mpl_toolkits.mplot3d` (for 3D PDPs).

Developer workflows
- Interactive: open and run notebooks in JupyterLab / Notebook at repo root:
  - `jupyter lab` or `jupyter notebook`
- Execute and export a single notebook programmatically (headless):
  - `jupyter nbconvert --to html <notebook>.ipynb --execute`
- Re-running notebooks may require creating a Python environment with the packages above. If you want, add a `requirements.txt` listing `pandas numpy matplotlib scikit-learn feature_engine shap`.

What AI agents should do (prioritized)
- When asked to edit code, prefer updating notebook cells in place (preserve narrative and output where possible).
- When adding helpers, create small Python modules (e.g. `src/helpers.py`) and import them from notebooks—avoid converting notebooks into a single script.
- When adding or changing dependencies, update a `requirements.txt` and include a short run example.
- Before adding SHAP code, verify `shap` is installed and available in the environment; otherwise, suggest adding it to the `requirements.txt`.

Files to check for examples
- `M6N3-Regression-house-prices.ipynb`: complete regression example using `Lasso`, `StandardScaler`, and `feature_engine`.
- `M6N2-Classification-titanic.ipynb`: classification pipeline with `feature_engine` and `GradientBoostingClassifier`.
- `M6N4_plot_permutation_importance.ipynb` & `M6N5_plot_partial_dependence.ipynb`: canonical model-interpretability patterns used in the course.

Notes & limitations
- This repository is educational: notebooks contain explanatory text and outputs — avoid destructive bulk refactors.
- Do not generate or assume CI/build scripts unless requested; there are no tests or build steps present.

If you want, I can:
- add a `requirements.txt` with discovered packages,
- add a short `README` with run commands,
- or expand examples in the instructions (for instance, exact `Pipeline` snippets pulled from notebooks).

Please tell me which of those (if any) you'd like next.
