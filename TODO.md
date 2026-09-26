# To-Do List

## 🚨 Critical — remaining before the API serves real predictions

- [ ] **Run the pipeline end-to-end.** All the machinery exists — what's left is *executing* it: `02_preprocessing.ipynb` (clean + combine + save processed splits), then either `03_modeling.ipynb` (TF-IDF + LogReg baseline) or the **BERTweet fine-tuning cells in `04_evaluation.ipynb`** (3 epochs, saves checkpoint to `outputs/models/bertweet-sexism/`).
- [ ] **Restart the API after training.** The lifespan handler auto-detects the fine-tuned checkpoint at `LOCAL_MODEL_DIR` and loads it instead of the fresh-headed `vinai/bertweet-base` — `/health` should report `"fresh_head": false`.
- [ ] ~~**Deploy model to HuggingFace Hub.**~~ Resolved differently: local inference is now the default backend (`INFERENCE_BACKEND=local`, model loaded at startup via FastAPI lifespan). If you prefer the HF Inference API instead, notebook `05` documents both routes: point `HF_MODEL` at `NLP-LTU/bertweet-large-sexism-detector` (already on the Hub) or upload your own baseline.

## 🐛 Bugs & repo hygiene

- [x] **Remove committed broken venv.** `.venv test/` (Linux binaries: `bin/python`, `lib64`, `pyvenv.cfg`) was accidentally committed. Deleted from git; `.venv*/` added to `.gitignore`.
- [x] **Remove junk file `data/processed/init`** (2-byte file; last commit was even titled "Rename s to init" — clearly accidental). Deleted.
- [x] **Add `python-dotenv` to `requirements.txt`.** `src/config/settings.py` calls `load_dotenv()` but the package wasn't listed — fresh installs crashed on import.
- [x] **Fix pydantic v2 deprecations in `src/api/fastapi_main.py`.** Migrated `@validator` → `@field_validator` with `@classmethod`.
- [x] **Fix import paths in notebooks.** Replaced the Colab-specific `sys.path.append('/content/...')` + `from data.load_data import ...` with a robust project-root setup and `from src.data.load_data import ...`.
- [x] **Move `02_preprocessing.ipynb`** from the repo root into `notebooks/` (01–05 structure now complete).
- [x] **Make `src/models/predict.py` lazy.** It used to `joblib.load()` at module import time (crash on fresh checkout); now loads on first use with a clear error message.
- [x] **Add `.env.example`.** Committed at the project root — the single env-var template (see dedup note below).
- [x] **Deduplicate configuration (single sources of truth).** Removed `ENV_TEMPLATE`/`create_env_template()` from `settings.py` (stale second copy of the env template); `hf_client.py` now imports `HF_API_KEY`/`HF_MODEL`/`MAX_TEXT_LENGTH`/`CACHE_SIZE` and logging config from `settings` instead of redefining them; library modules no longer call `logging.basicConfig`. All 32 settings env vars are documented in `.env.example`.
- [x] **Fix config default.** `DATASET_PATH` now defaults to `data/raw/dev.csv` (was the nonexistent `sexism_dataset.csv`); `.env.example` and `settings.py` agree.
- [x] **Track `outputs/` scaffolding.** `outputs/{models,logs,reports,inference}` committed as `.gitkeep` placeholders (contents still gitignored) so fresh clones match the README structure.
- [ ] **Unify label handling.** Dataset labels are `"not sexist"` / `"sexist"` (with spaces), but `LABEL_MAP` in settings still uses `"not_sexist"` / `"sexist"` — align them.
- [ ] **CORS `allow_origins=["*"]` with `allow_credentials=True`** is an insecure combination — restrict origins before any real deployment.

## 🔧 Improvements to existing code

- [x] **Improve `train.py`:** stratified held-out split, `class_weight="balanced"`, classification report on the held-out set, settings-driven paths/features (`TFIDF_MAX_FEATURES`, ngram range).
- [x] **Get train/test split files.** Solved via `src/data/load_kaggle.py` — `load_edos_splits()` downloads the three EDOS splits through `kagglehub` (same source as the coursework notebook); the raw CSVs also exist in the Drive `Sexism-Classification` folder if you prefer committing them.
- [x] **Restore the full pipeline into the canonical notebooks.** The complete coursework pipeline (X3a7e2118_1_066.ipynb) now lives in `01`–`05`: EDA, pandera + domain-aware cleaning, 10-fold CV over LR/NB/RF/SVM (03), CNN/LSTM/BERTweet/SMOTE/SHAP (04), HF export paths (05).
- [x] **Local inference with lifespan.** Model loads at startup via FastAPI's `lifespan` (`src/api/model_manager.py`): downloads `vinai/bertweet-base` weights with `AutoModelForSequenceClassification`, attaches a fresh classification head, and serves softmax probabilities; fine-tuned checkpoint takes priority when present.
- [ ] **Ensure train/inference preprocessing parity.** The fine-tuning cells in `04` train on *cleaned* text (`clean_text` from `02`), but `model_manager.predict()` sends *raw* text to BERTweet. Either fine-tune on raw text, or apply the same cleaning in `model_manager` — pick one and make both paths match.
- [ ] **Use full label set or document binary-only choice.** EDOS has hierarchical labels (`label_sexist`, `label_category`, `label_vector`) — currently everything downstream only supports binary. Decide: binary only, or add category/vector classification.
- [ ] **Uncomment/decide on optional deps** (redis, slowapi, prometheus) — or remove the dead `CACHE_TTL` / rate-limit promises from the README if not implementing.

## 🧪 Infrastructure

- [x] **Test suite.** 20 tests: API endpoints with a mocked model manager (hermetic — `PRELOAD_MODEL=false`, no real weights downloaded), preprocessing utilities, domain-aware cleaning.
- [x] **CI (GitHub Actions).** Runs `pytest tests/ -v` on push/PR across Python 3.11 and 3.12; badge in README.
- [x] **Dockerfile.** Slim Python image, non-root user, healthcheck; model downloads at container start (or bake the checkpoint into the image for offline deploys).
- [x] **README performance section.** Points at this TODO until the first real evaluation run; then fill in the numbers.

## 📌 Remaining order of attack

1. Run `02_preprocessing.ipynb` → produce processed splits
2. Run the BERTweet fine-tuning cells in `04_evaluation.ipynb` (GPU recommended)
3. Restart the API → verify `/health` shows `fresh_head: false` → `/predict` serves your model
4. Decide label handling (binary vs full EDOS hierarchy) + preprocessing parity
5. Optional: HF Hub deployment (notebook 05), optional deps decision
