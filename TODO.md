# To-Do List

## 🚨 Critical — project doesn't work without these

- [ ] **Train an actual model.** No trained model exists (`outputs/models/` is empty and gitignored). `src/models/train.py` exists but expects `data/processed/train.csv`, which was never produced. Full pipeline needed: raw data → preprocess → split → train → evaluate → save.
- [ ] **Get train/test split files.** Only `data/raw/dev.csv` is in the repo — the EDOS 2022 train and test CSVs are missing. Commit the splits (or add a download script).
- [ ] **Deploy model to HuggingFace Hub.** `HF_MODEL` in settings is still the placeholder `"your-model-name"`. The FastAPI service calls the HF Inference API, so it cannot work until a model is pushed to the Hub and configured via `.env`.
- [ ] **Finish preprocessing.** `02_preprocessing.ipynb` stops after loading data + pandera validation + NLTK setup — no actual cleaning/splitting has been run or saved.

## 🐛 Bugs & repo hygiene

- [x] **Remove committed broken venv.** `.venv test/` (Linux binaries: `bin/python`, `lib64`, `pyvenv.cfg`) was accidentally committed. Delete from git (`git rm -r ".venv test"`) and add `.venv*/` to `.gitignore`.
- [x] **Remove junk file `data/processed/init`** (2-byte file; last commit was even titled "Rename s to init" — clearly accidental). Delete it.
- [x] **Add `python-dotenv` to `requirements.txt`.** `src/config/settings.py` calls `load_dotenv()` but the package isn't listed — fresh installs crash on import.
- [x] **Fix pydantic v2 deprecations in `src/api/fastapi_main.py`.** `@validator` is deprecated (removed in pydantic v2.10+) — migrate to `@field_validator` with `@classmethod`. Also review `class Config` usage.
- [x] **Fix import paths in notebooks.** Both notebooks do `sys.path.append('/content/Sexism-Classification/src')` (Colab-specific) and import `from data.load_data import ...` — should be `from src.data.load_data import ...` so they run locally too.
- [x] **Move `02_preprocessing.ipynb`** from the repo root into `notebooks/` (README says it lives there).
- [ ] **Unify label handling.** Dataset labels are `"not sexist"` / `"sexist"` (with spaces), but `LABEL_MAP` in settings uses `"not_sexist"` / `"sexist"` — align them.
- [x] **Make `src/models/predict.py` lazy.** It calls `joblib.load()` at module import time, so importing it crashes if no model exists yet. Load inside the predict function with a clear error message.
- [x] **Add `.env.example`.** README references it; `settings.py` can generate it — create and commit it.

## 🔧 Improvements to existing code

- [x] **Improve `train.py`:** add stratified train/val/test split, class-imbalance handling (dataset is imbalanced — check label distribution), `class_weight="balanced"`, evaluation on a held-out set (not just training accuracy), and configurable paths via `settings.py`.
- [ ] **Use full label set or document binary-only choice.** EDOS has hierarchical labels (`label_sexist`, `label_category`, `label_vector`) — currently everything downstream only supports binary. Decide: binary only, or add category/vector classification.
- [ ] **Ensure train/inference preprocessing parity.** API's `hf_client._preprocess_text()` and training's `preprocess.clean_text()` do different things — predictions will be wrong if training cleaned text one way and the API cleans it another. Use the same pipeline in both.
- [ ] **Fix config default.** `DATASET_PATH` defaults to `data/raw/sexism_dataset.csv`, which doesn't exist — point it at the real file(s).
- [ ] **CORS `allow_origins=["*"]` with `allow_credentials=True`** is an insecure combination — restrict origins before any real deployment.

## 📓 Missing notebooks (README promises 5, repo has 2)

- [ ] `notebooks/03_modeling.ipynb` — baseline LogReg/TF-IDF, SVM, then BERT fine-tune
- [ ] `notebooks/04_evaluation.ipynb` — accuracy/precision/recall/F1, confusion matrix, error analysis
- [ ] `notebooks/05_export_model.ipynb` — export to HF format and push to the Hub

## 🧪 Missing infrastructure

- [x] **Test suite.** README references `pytest tests/` and `test_api.py` — neither exists. At minimum: unit tests for preprocessing, API endpoint tests with mocked HF client (`httpx` + FastAPI `TestClient`).
- [x] **Dockerfile.** README documents `docker build` but no Dockerfile exists in the repo.
- [ ] **CI** (GitHub Actions): lint + run tests on push.
- [x] **Fill in the Model Performance section of the README** — currently says `Accuracy: XX%`. Update after the first real evaluation run.
- [ ] **Uncomment/decide on optional deps** (redis, slowapi, prometheus) — or remove the dead `CACHE_TTL` / rate-limit promises from the README if not implementing.

## 📌 Suggested order of attack

1. Repo hygiene (delete venv + junk file, fix requirements, pydantic, imports) — quick wins
2. Data: get train/test CSVs, run preprocessing, produce splits
3. Train baseline (LogReg + TF-IDF) → evaluate → fill in README metrics
4. Deploy something to HF Hub → get the API working end-to-end
5. Missing notebooks 03–05
6. Tests + Dockerfile + CI
