# To-Do List

## ✅ Done — baseline pipeline runs end-to-end

- [x] **Baseline model trained on real EDOS data.** `outputs/models/classifier.joblib` + `vectorizer.joblib` — TF-IDF + Logistic Regression (`class_weight="balanced"`), trained on 14k rows. Held-out evaluation: **accuracy 0.79, weighted F1 0.80** (sexist class F1 0.61).
- [x] **Data acquisition.** `src/data/load_kaggle.py` — `load_edos_splits()` downloads all three EDOS splits via kagglehub; raw CSVs committed to `data/raw/` (train/dev/test, 20k rows).
- [x] **Processed splits committed.** `data/processed/{train,dev,test}.csv` (cleaned text + binary labels, 2.6 MB) — cloned repos can train without re-downloading.
- [x] **Evaluation artifacts.** `outputs/reports/`: `evaluation.json` (test-set n=4,000), `classification_report.txt`, `confusion_matrix.png`, `label_distribution.png`, `top_bigrams_{sexist,not_sexist}.csv`.
- [x] **Inference demo.** `outputs/inference/predictions.csv` — works, but exposes the baseline's weakness: "Get back to the kitchen where you belong" → *not sexist* (0.76). This is the gap the BERTweet fine-tuning closes.
- [x] **Training + evaluation logs.** `outputs/logs/{training_log,evaluation_log}.txt`.
- [x] **Local inference with lifespan.** Model loads at startup via FastAPI's `lifespan` (`src/api/model_manager.py`): downloads `vinai/bertweet-base` weights with `AutoModelForSequenceClassification`, attaches a fresh classification head, serves softmax probabilities; fine-tuned checkpoint takes priority when present.
- [x] **HF-standard token naming.** Single `HF_TOKEN` (accepts `HUGGING_FACE_HUB_TOKEN`; legacy `HF_API_KEY` fallback) replaces the old `HF_API_KEY`/`HF_TOKEN` split. Renamed across settings, hf_client, `.env.example`, CI, tests, notebook 05.
- [x] **Optional API-key auth.** `API_SECRET_KEY` + `X-API-Key` header middleware (403 on missing/wrong key; `/`, `/health`, `/docs` stay public). Verified live: no header → 403, correct key → 200.
- [x] **Deduplicate configuration (single sources of truth).** Removed `ENV_TEMPLATE`/`create_env_template()` from `settings.py`; `hf_client.py` imports `HF_MODEL`/`HF_TOKEN`/constants from settings; logging uses `settings.LOG_LEVEL/LOG_FORMAT`. All settings env vars documented in `.env.example`.
- [x] **Restore the full pipeline into the canonical notebooks.** Coursework pipeline (X3a7e2118_1_066.ipynb) distributed into `01`–`05`: EDA, pandera + domain-aware cleaning, 10-fold CV over LR/NB/RF/SVM (03), CNN/LSTM/BERTweet/SMOTE/SHAP (04), HF export paths (05).
- [x] **Improve `train.py`:** stratified held-out split, `class_weight="balanced"`, classification report on held-out data, settings-driven paths/features.
- [x] **Test suite.** 20 tests: API endpoints with a mocked model manager (hermetic — `PRELOAD_MODEL=false`), preprocessing, domain-aware cleaning.
- [x] **CI (GitHub Actions).** `pytest tests/ -v` on push/PR across Python 3.11/3.12; badge in README.
- [x] **Dockerfile.** Slim Python image, non-root user, healthcheck.
- [x] **Repo hygiene.** Broken `.venv test/` and junk files removed; `02_preprocessing.ipynb` moved into `notebooks/`; `outputs/` scaffolding tracked; stale clone + stray debug artifacts (`cell_report.txt`, `full_output.json`) deleted; pydantic v2 migration; `python-dotenv` dependency fix; lazy `predict.py`; notebook import paths fixed; `.env.example` committed; README structure now matches reality (project renamed `sexism-classification`).

## 🚨 Remaining — to get real (non-baseline) predictions

- [ ] **Run BERTweet fine-tuning (GPU).** Notebook `04_evaluation.ipynb` has the complete cells: `AutoModelForSequenceClassification` from `vinai/bertweet-base` + fresh head, 3 epochs on the EDOS train split (`lr=2e-5`, `batch=16`, `max_length=128`), saves the checkpoint to `outputs/models/bertweet-sexism/`. CPU works but is slow — Colab T4 recommended.
- [x] **Restart the API after training.** The lifespan auto-detects the checkpoint at `LOCAL_MODEL_DIR` and loads it instead of the fresh-headed base — `/health` should report `"fresh_head": false`, and the edge cases the baseline misses (e.g. "Get back to the kitchen where you belong") should flip to *sexist*.
- [ ] **Ensure train/inference preprocessing parity.** The fine-tuning cells in `04` train on *cleaned* text (`clean_text` from `02`), but `model_manager.predict()` sends *raw* text to BERTweet. Pick one side and align both paths.
- [x] **Unify label handling.** Dataset labels are `"not sexist"` / `"sexist"` (with spaces); `LABEL_MAP` in settings still uses `"not_sexist"` / `"sexist"`.
- [ ] **CORS `allow_origins=["*"]` with `allow_credentials=True`** is an insecure combination — restrict origins before any real deployment.
- [ ] **Use full label set or document binary-only choice.** EDOS has hierarchical labels (`label_sexist`, `label_category`, `label_vector`) — decide: binary only, or add category/vector classification.
- [ ] **Uncomment/decide on optional deps** (redis, slowapi, prometheus) — or remove the dead `CACHE_TTL` / rate-limit promises from the README if not implementing.

## 📌 Runbook (from clean clone to working API)

```bash
git clone https://github.com/Oluchi-Otuadinma/Sexism-Classification.git
cd Sexism-Classification
pip install -r requirements.txt

pytest tests/ -v                      # 20 tests, sanity check
uvicorn src.api.fastapi_main:app      # baseline API (local backend, no token needed)

# then, for the real model:
#   run notebooks 02 -> 04 (fine-tune) -> restart API -> /health shows fresh_head: false
```
