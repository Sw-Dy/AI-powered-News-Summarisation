<a name="readme-top"></a>
<br />
<div align="center">
  <img src="https://github.com/Oxlac/AI-News-Summariser/assets/73121234/35981902-47c3-4143-ad88-6ff1100e9c64" alt="Logo" width="200" height="200">
  <h3 align="center">AI-News-Summariser</h3>
  <p align="center">
    A tool for automatically summarizing news articles using artificial intelligence.
    <br />
    <a href="https://aisummariser.oxlac.com">Visit the Website »</a>
    <br />
    <br />
    <a href="https://www.aadinagarajan.com/#contact">Contact Developer</a>
    ·
    <a href="https://github.com/Oxlac/AI-News-Summariser/issues">Report Bug</a>
    ·
    <a href="https://github.com/Oxlac/AI-News-Summariser/issues">Request Feature</a>
    .
    <a href="https://discord.gg/x3ba4sTzgd">Discord Support</a>
  </p>
</div>

## About The Project

![AI-News-Summariser Screen Shot](image.png)

AI-News-Summariser is an end-to-end summarization system for news articles and uploaded files. It provides:

- A Flask web UI and API for summarizing URLs, raw text, and file uploads.
- A TF-IDF extractive baseline used as a fast default and as a filter for the TF-IDF + PGN hybrid model.
- Optional abstractive backends (T5, DistilBART, BART, PEGASUS).
- A research-grade TF-IDF guided Pointer-Generator Network (PGN) implemented in PyTorch with coverage and extended vocabulary.
- A browser extension that can summarize the current page via the local API.
- A video/audio pipeline that transcribes media and summarizes the transcription.

>[!CAUTION]
>Ensure that you use this tool responsibly. Respect the copyrights and terms of use of the news sources.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Table of Contents

- [System Overview](#system-overview)
- [TF-IDF + PGN Research Deep Dive](#tf-idf--pgn-research-deep-dive)
- [Methodology and Data Flow](#methodology-and-data-flow)
- [Training and Evaluation](#training-and-evaluation)
- [API Surface](#api-surface)
- [Configuration and Settings](#configuration-and-settings)
- [Repository Structure and File Map](#repository-structure-and-file-map)
- [Getting Started](#getting-started)
- [Limitations](#limitations)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## System Overview

At runtime the system follows a hybrid decision path:

1. Input ingestion:
   - URL → fetch, parse, and clean article text.
   - Upload → read text/PDF/OCR or transcribe video/audio.
2. Summarization:
   - Default and fallback model selection (TF-IDF, TF-IDF + PGN, or pretrained abstractive).
   - Extractive chunking for long inputs.
3. Analysis:
   - Optional sentiment, key phrases, named entities, readability score, and Q/A pair.
4. Persistence:
   - Results are stored in SQLite for history and settings.

The TF-IDF baseline is both a standalone summarizer and a first-stage filter for the TF-IDF + PGN pipeline. This makes it easy to compare extractive vs. hybrid summaries and isolate the effect of abstractive modeling on a reduced input.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## TF-IDF + PGN Research Deep Dive

This repository implements a two-stage hybrid summarizer:

- Stage 1 (Extractive): A frozen TF-IDF sentence ranker selects the top-k sentences from the article.
- Stage 2 (Abstractive): A Pointer-Generator Network (PGN) summarizes the TF-IDF-filtered text.

The design intentionally removes pretrained language models to highlight the contribution of explicit extractive filtering and pointer-based copying. This aligns with low-resource and CPU-friendly settings.

### Stage 1: TF-IDF Sentence Ranking

**Vectorizer training**
- The extractive model is a `TfidfVectorizer` trained on the dataset corpus.
- Configuration is stored in `outputs/cnn_dm_extractive/model_config.json`.
- Key parameters from `ml/train.py`:
  - `stop_words="english"`
  - `max_features` (default 50,000)
  - `min_df` (default 2)
  - `max_train_samples` (default 5,000)

**Core scoring**
- The input is split into sentences; each sentence is transformed into a TF-IDF vector.
- Sentence score = sum of TF-IDF weights across all features in that sentence vector.
- Top-k sentences are selected by score, then re-ordered to preserve narrative flow.

**Formalization**

Let \( d \) be a sentence, \( t \) a token, and \( \mathcal{V} \) the vocabulary.

- TF component: \( \mathrm{tf}(t, d) = \text{count}(t, d) \)
- IDF component: \( \mathrm{idf}(t) = \log\left(\frac{N + 1}{\mathrm{df}(t) + 1}\right) + 1 \)
- TF-IDF: \( \mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t) \)
- Sentence score: \( S(d) = \sum_{t \in \mathcal{V}} \mathrm{tfidf}(t, d) \)

The implementation uses scikit-learn defaults for terms not explicitly configured.

### Stage 2: Pointer-Generator Network (PGN)

The PGN is implemented in `ml/pgn.py` and uses:

- Encoder: single-layer BiLSTM.
- Decoder: single-layer LSTM with Bahdanau attention.
- Coverage: tracks cumulative attention to reduce repetition.
- Extended vocabulary: enables copy from source tokens.

**Mixture distribution**

At each decoding step:

\[
P(\text{word}) = p_{gen} \cdot P_{\text{vocab}}(\text{word}) + (1 - p_{gen}) \cdot P_{\text{copy}}(\text{word})
\]

Where:

- \( P_{\text{vocab}} \) is the decoder vocabulary distribution.
- \( P_{\text{copy}} \) is the attention distribution projected into the extended vocabulary.
- \( p_{gen} \in [0, 1] \) is computed from context, decoder state, and input embedding.

**Coverage**

Coverage is tracked as:

\[
cov_t = \sum_{i=0}^{t-1} a_i
\]

The loss uses coverage penalty:

\[
L = -\log P(y_t) + \lambda \sum_j \min(a_t^j, cov_t^j)
\]

Where \( a_t \) is the attention distribution at step \( t \).

**Extended vocabulary**

For each input:
- All OOV tokens are collected into an OOV list.
- IDs are mapped into an extended vocabulary: base vocab + per-example OOV slots.
- The final distribution scatters copy probabilities to extended indices.

### Why TF-IDF Guidance

The TF-IDF filter:

- Shrinks long articles to a stable input budget.
- Reduces exposure to boilerplate or low-signal sentences.
- Makes PGN training more sample-efficient by focusing on salient input.

This is explicitly encoded in the dataset builder (`build_tfidf_dataset`) that maps each article to a TF-IDF summary before PGN training.

### Reported Results and Limitations

The repository includes a research summary in `outputs/pgn_report.md`:

- ROUGE scores are provided for a smoke-test run.
- The report emphasizes that strong ROUGE needs full-scale training.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Methodology and Data Flow

### Ingestion and Parsing

- URLs are fetched with custom headers and timeouts.
- `newspaper3k` parses article content, title, authors, and top image.
- If parsing fails, the system falls back to HTML title extraction and text stripping.

### Preprocessing

- `normalize_text` collapses whitespace and normalizes Unicode spaces.
- Sentence segmentation uses NLTK if available, otherwise regex-based splitting.
- Chunking by sentence length allows long documents to be summarized in parts.

### Summarization Path

1. Select model:
   - The UI settings decide `selectedModel` and `fallbackModel`.
2. Compute summary:
   - `tfidf` → extractive ranking.
   - `tfidf-pgn` → extractive filter + PGN decoding.
   - `t5`, `bart`, `pegasus` → abstractive transformer pipeline.
3. Post-process:
   - Optional output format (paragraph, bullets, TL;DR).
   - Optional language style normalization for simpler phrasing.

### Analysis Layer

- Key phrases extracted by high-frequency bigrams (stopword filtered).
- Named entities extracted via capitalization heuristic.
- Readability computed with a Flesch-like formula.
- Simple Q/A pair generated from the first sentence.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training and Evaluation

### TF-IDF Baseline Training

- Train vectorizer with `ml/train.py` on CNN/DailyMail (or a local dataset).
- Writes `model.pkl` and `model_config.json` under `outputs/cnn_dm_extractive`.

### PGN Training

- `ml/pgn.py` builds the TF-IDF filtered dataset.
- SentencePiece BPE is trained from scratch on TF-IDF inputs and gold summaries.
- Training uses Adam, gradient accumulation, dropout, coverage loss, and checkpointing.
- Evaluation computes ROUGE for both TF-IDF baseline and PGN outputs.

### Evaluation Pipeline

- `evaluate_rouge` builds the TF-IDF dataset, runs PGN decoding, and compares against gold summaries.
- ROUGE is computed for both PGN and TF-IDF baseline to establish relative gains.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## API Surface

The Flask server exposes:

- `GET /` → Web UI
- `POST /api/summarize` → Summarize URL or raw text
- `POST /api/summarize-upload` → Summarize files or media
- `GET/POST /api/settings` → Persist settings
- `GET /api/history` → Summary history

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration and Settings

Settings are stored in SQLite and merged with defaults at runtime:

- Model selection: default, fallback, auto routing.
- Length control: short/medium/long mapped to sentence count.
- Output format: paragraph, bullets, TL;DR.
- Performance: max input length, chunk size, streaming flag.
- Analysis: toggles for sentiment, key phrases, entities, readability, Q/A.

Runtime environment variables include:

- `MODEL_DIR`, `TRAIN_CONFIG`, `AUTO_TRAIN`
- `PGN_CHECKPOINT`, `PGN_TOKENIZER`
- `PGN_BEAM_SIZE`, `PGN_MAX_SUMMARY_LEN`
- `PGN_COVERAGE_PENALTY`, `PGN_LENGTH_PENALTY`, `PGN_NO_REPEAT_NGRAM_SIZE`
- `PGN_MAX_SOURCE_LEN`, `PGN_HIDDEN_SIZE`, `PGN_EMBED_SIZE`, `PGN_DROPOUT`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure and File Map

This section documents every file and its role.

### Root

- `app.py`: Flask server, routing, summarization orchestration, SQLite persistence, settings handling, analysis utilities, and API responses.
- `requirements.txt`: Python dependencies (Flask, NLTK, scikit-learn, transformers, datasets, sentencepiece, etc.).
- `README.md`: This document.
- `TEN_WEEK_PROJECT_PLAN.md`: Long-form project plan and proposed architecture.
- `summarizer.log`: Runtime log output from the server.
- `.gitattributes`: Git attribute settings for the repository.

### ML Pipeline (`ml/`)

- `ml/train.py`: Trains TF-IDF vectorizer on dataset and writes `model.pkl` + metadata.
- `ml/summarizer.py`: Loads TF-IDF model and performs extractive sentence ranking.
- `ml/data_utils.py`: Normalization, token chunking, hashing/deduplication utilities.
- `ml/pgn.py`: Full PGN implementation with SentencePiece, dataset building, training, beam search, coverage, and ROUGE evaluation.
- `ml/console_backend.py`: CLI summarizer for URL input with optional PGN decoding.
- `ml/configs/accelerate_single_gpu.yaml`: Accelerate configuration for training.
- `ml/configs/cnn_dm_bart.json`: Example configuration for abstractive training.
- `ml/__init__.py`: Package marker.

### Services (`services/`)

- `services/video_summarizer.py`: Video/audio pipeline that extracts audio, transcribes with ASR, and summarizes via extractive/abstractive backends. Includes validation and error classes.

### Outputs (`outputs/`)

- `outputs/cnn_dm_extractive/model.pkl`: Trained TF-IDF vectorizer.
- `outputs/cnn_dm_extractive/model_config.json`: Metadata for the TF-IDF model.
- `outputs/pgn_report.md`: Research summary and ROUGE results for TF-IDF + PGN.
- `outputs/pgn/`: SentencePiece models and training corpus for PGN.
- `outputs/pgn_smoke/`: Smoke-test checkpoint, ROUGE results, and training stats.
- `outputs/pgn_smoke2/`: Multi-epoch smoke-test checkpoints and metrics.

### Data (`data/`)

- `data/app.db`: SQLite database for summaries and settings.
- `data/hf_datasets/cnn_dailymail_3.0.0/`: Cached HF dataset shards (train/validation/test).
- `data/hf_datasets/cnn_dailymail_3.0.0/dataset_dict.json`: Split metadata.

### Frontend (`templates/`, `static/`)

- `templates/index.html`: Main UI for URL/file summarization and results.
- `templates/settings.html`: Settings UI.
- `templates/history.html`: Summary history UI.
- `static/app.js`: Frontend orchestration, summarization calls, UI motion, and results rendering.
- `static/history.js`: History API calls and rendering.
- `static/settings.js`: Settings UI wiring and persistence calls.
- `static/dark-theme.css`: Dark theme styles and animations.
- `static/light-theme.css`: Light theme styles and animations.
- `static/settings.css`: Settings page styles.
- `static/logo.jpeg`: UI logo asset.
- `static/tumbnail.png`: UI thumbnail asset.
- `static/favicon_io/android-chrome-192x192.png`: Favicon assets.
- `static/favicon_io/android-chrome-512x512.png`: Favicon assets.
- `static/favicon_io/apple-touch-icon.png`: Favicon assets.
- `static/favicon_io/favicon-16x16.png`: Favicon assets.
- `static/favicon_io/favicon-32x32.png`: Favicon assets.
- `static/favicon_io/favicon.ico`: Favicon assets.
- `static/favicon_io/site.webmanifest`: Favicon manifest.

### Core JS Modules (`core/`)

- `core/modelRouter.js`: Smart routing for model selection by input length.
- `core/settingsManager.js`: Settings persistence, API sync, and default model list.
- `core/summarizer.js`: Frontend API wrapper for /api/summarize and formatting.
- `core/analyzer.js`: Lightweight frontend analysis helpers.

### Chrome Extension (`chrome_extension/`)

- `chrome_extension/manifest.json`: Extension manifest and permissions.
- `chrome_extension/popup.html`: Extension UI.
- `chrome_extension/popup.css`: Extension styling.
- `chrome_extension/popup.js`: Extracts page text and calls local API.
- `chrome_extension/core/modelRouter.js`: Extension copy of model routing logic.
- `chrome_extension/core/settingsManager.js`: Extension settings storage.
- `chrome_extension/core/summarizer.js`: Extension API wrapper for summarization.

### GitHub Templates (`.github/`)

- `.github/ISSUE_TEMPLATE/bug_report.md`: Bug report template.
- `.github/ISSUE_TEMPLATE/feature_request.md`: Feature request template.

### Cache and Build Artifacts

- `.ruff_cache/`: Ruff lint cache.
- `__pycache__/`: Python bytecode cache.
- `ml/__pycache__/`: ML module bytecode cache.
- `services/__pycache__/`: Services module bytecode cache.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Complete Workflow and Function Map

This section documents the full end-to-end workflow, the model-selection rules, and every function/class in the current codebase.

### End-to-End Workflow

1. User selects input type and model settings:
   - Web UI and extension allow choosing a default model and fallback model.
   - Settings include summary length, output format, and analysis toggles.
2. Frontend sends request:
   - `core/summarizer.js` or `chrome_extension/core/summarizer.js` calls `/api/summarize`.
   - If a file is uploaded, `/api/summarize-upload` is used instead.
3. Backend resolves settings:
   - Defaults merge with persisted settings from SQLite.
   - Model selection resolves to a concrete model ID.
4. Text ingestion:
   - URL → `fetch_url` + `parse_article` (or fallback HTML extraction).
   - Uploads → `extract_text_from_file` or video/audio transcription.
5. Summarization execution:
   - `tfidf` uses extractive ranking.
   - `tfidf-pgn` uses extractive filtering then PGN decoding.
   - Transformer models use the Hugging Face pipeline.
6. Post-processing and analysis:
   - Output formatting (paragraph/bullets/TL;DR).
   - Optional sentiment, key phrases, named entities, readability, Q/A.
7. Persistence:
   - Summary and settings are stored in SQLite.
8. Response:
   - JSON payload returned to UI/extension and rendered in the UI.

### Model Selection Rules

- Frontend routing (`core/modelRouter.js`, `chrome_extension/core/modelRouter.js`):
  - If a user selects a fixed model, it is used.
  - Otherwise, routing is based on input length:
    - ≤ 1200 → `t5-small`
    - ≤ 4000 → `distilbart`
    - ≤ 8000 → `bart-large`
    - else → fallback (default `tfidf-pgn`)
- Backend routing (`app.py`):
  - `select_model_id` returns `selectedModel` or `defaultModel`.
  - If `auto`, it uses the fallback model (`tfidf-pgn` by default).
  - `summarize_with_model` executes primary then fallback if needed.

### Backend Workflow (Flask)

- `app.py` orchestrates the API and model lifecycle:
  - Loads TF-IDF vectorizer at startup.
  - Initializes the SQLite schema.
  - Exposes UI routes and API routes.
  - Manages summarization, analysis, and persistence.

### Function and Class Reference

#### app.py

- `deep_merge`: recursive dict merge for settings overlays.
- `build_full_settings`: merges payload with defaults.
- `resolve_settings`: merges stored settings with payload.
- `allowed_file`: extension whitelist for uploads.
- `extract_text_from_file`: reads `.txt`, `.pdf`, OCR for images.
- `ensure_db_dir`: creates SQLite directory if missing.
- `get_db_connection`: opens SQLite connection with row factory.
- `init_db`: creates `summaries` and `settings` tables.
- `load_settings_from_db`: reads settings JSON from SQLite.
- `save_settings_to_db`: upserts settings JSON.
- `record_summary`: writes summary history to SQLite.
- `fetch_history`: reads latest summaries for history view.
- `model_is_ready`: checks presence of TF-IDF model.pkl.
- `train_model`: runs `ml.train` using configured JSON.
- `get_website_name`: extracts domain for fallback authors.
- `normalize_url`: ensures URL has a scheme.
- `fetch_url`: requests URL with default headers.
- `cors_json`: JSON response with CORS headers.
- `summary_length_to_sentences`: maps short/medium/long to sentence count.
- `sentence_split`: NLTK sentence tokenizer with regex fallback.
- `chunk_text_by_sentences`: splits long text into sentence chunks.
- `summarize_extractive_with_chunks`: chunk-aware TF-IDF summary.
- `extract_key_phrases`: frequency-based phrase extractor.
- `extract_named_entities`: capitalization-based entity extractor.
- `count_syllables`: syllable counter for readability.
- `compute_readability`: Flesch-like readability score.
- `apply_language_style`: simple style normalization for readability.
- `apply_output_format`: paragraph/bullets/TL;DR formatting.
- `build_qa_pairs`: simple Q/A pair from first summary sentence.
- `select_model_id`: resolves chosen model from settings.
- `get_fallback_model`: returns fallback model from settings.
- `get_abstractive`: loads transformer model + tokenizer.
- `summarize_abstractive`: chunked transformer summarization.
- `pgn_args`: resolves PGN decoding configuration from env.
- `get_pgn`: loads PGN checkpoint and tokenizer.
- `summarize_with_pgn`: PGN beam search decoding on TF-IDF text.
- `is_gibberish`: heuristic filter for invalid summaries.
- `summarize_with_model`: primary + fallback orchestration.
- `parse_article`: newspaper3k parsing, metadata extraction.
- `extract_title_from_html`: HTML title fallback.
- `extract_text_from_html`: HTML text fallback.
- `analyze_sentiment`: TextBlob sentiment classification.
- `index`: server-rendered web UI handler.
- `settings`: settings page view.
- `history`: history page view.
- `core_assets`: serves `/core/` JS modules.
- `api_summarize`: JSON API for URL/text summarization.
- `api_summarize_upload`: JSON API for file summarization.
- `api_settings`: settings API (GET/POST).
- `api_history`: history API (GET).

#### ml/data_utils.py

- `set_seed`: seeds Python, NumPy, and Torch.
- `normalize_text`: whitespace and unicode normalization.
- `hash_text`: SHA1 hash for dedup.
- `simhash`: SimHash fingerprint for dedup.
- `hamming_distance`: SimHash distance check.
- `deduplicate_dataset`: dataset-level dedup filter.
- `chunk_text_by_tokens`: tokenizer-based chunking.
- `ensure_dir`: directory creation helper.

#### ml/summarizer.py

- `load_extractive_model`: loads `model.pkl` and config.
- `summarize_extractive`: TF-IDF sentence ranking and selection.

#### ml/train.py

- `build_parser`: CLI args for TF-IDF training.
- `parse_args`: config file + CLI merge.
- `get_dataset_fields`: dataset column mapping.
- `load_data`: HF dataset loading.
- `collect_texts`: gathers training texts for vectorizer.
- `main`: trains and serializes TF-IDF model.

#### ml/pgn.py

- `train_sentencepiece`: trains SentencePiece BPE.
- `clean_text_for_spm`: pre-tokenization text cleanup.
- `build_tfidf_dataset`: TF-IDF-guided dataset builder.
- `SpTokenizer`: wrapper around SentencePiece.
- `build_extended_ids`: builds extended vocab IDs + OOV list.
- `map_target_ids`: maps target IDs into extended space.
- `PgnDataset`: dataset wrapper for PGN training.
- `collate_fn`: batch padding and tensor collation.
- `Encoder`: BiLSTM encoder.
- `BahdanauAttention`: additive attention with coverage.
- `PGNDecoder`: LSTM decoder with p_gen computation.
- `PointerGenerator`: full PGN model with coverage loss.
- `calc_final_dist`: combines vocab and copy distributions.
- `TrainConfig`: training hyperparameter dataclass.
- `build_train_config`: builds config from args.
- `ensure_dir`: re-exported helper for output directory.
- `set_seed`: re-exported helper for reproducibility.
- `load_hf_dataset`: dataset loader for PGN training.
- `write_corpus_file`: writes SentencePiece corpus.
- `save_checkpoint`: writes PGN checkpoint.
- `train_model`: PGN training loop with coverage loss.
- `evaluate_loss`: validation loss calculation.
- `beam_search`: PGN beam decoding with coverage penalty.
- `decode_tokens`: converts IDs to text with OOV handling.
- `generate_summary`: helper for decoding one item.
- `evaluate_rouge`: ROUGE evaluation for PGN vs TF-IDF.
- `sanity_check`: quick inference sanity test.
- `report_resources`: training resource stats.
- `dir_size`: output directory size estimator.

#### ml/console_backend.py

- `parse_args`: CLI argument parsing for console use.
- `fetch_article`: URL fetch + parse with newspaper3k.
- `load_pgn`: load PGN checkpoint and tokenizer.
- `summarize_with_pgn`: decoding helper for CLI.
- `is_gibberish`: heuristic filter for invalid summaries.
- `main`: CLI flow for TF-IDF or TF-IDF + PGN.

#### services/video_summarizer.py

- `VideoProcessingError`, `AudioExtractionError`, `TranscriptionFailedError`, `SummarizationError`: error classes.
- `VideoSummarizer.__init__`: configuration loading.
- `summarize_video`: full video pipeline entrypoint.
- `transcribe_video`: audio extraction + ASR.
- `extract_audio`: ffmpeg extraction and validation.
- `transcribe_audio`: chunked ASR inference + confidence.
- `clean_text`: ASR cleanup and sentence normalization.
- `summarize_text`: model selection for transcripts.
- `_summarize_extractive`: TF-IDF fallback summarization.
- `_summarize_abstractive`: transformer summarization for long text.
- `_prepare_extractive_text`: chunking and sentence formatting.
- `_get_extractive_vectorizer`: lazy TF-IDF vectorizer load.
- `_get_abstractive`: lazy transformer load.
- `_get_asr`: lazy ASR load.
- `_load_audio`, `_normalize_audio`, `_trim_silence`, `_chunk_audio`: preprocessing helpers.
- `_logit_confidence`: confidence scoring for ASR output.
- `_is_transcription_valid`: validation gate for ASR.
- `_get_audio_duration`: duration metadata extraction.
- `_get_device`: device selection.
- `_no_grad`: torch inference context manager.

#### core/analyzer.js

- `analyzeText`: lightweight UI analysis helper (word count, heuristics).

#### core/modelRouter.js

- `selectModel`: selects a model based on length and user preference.
- `applyFallback`: applies fallback when primary is missing.

#### core/settingsManager.js

- `getSettings`: loads settings (local storage + API).
- `saveSettings`: persists merged settings.
- `resetSettings`: reset to defaults.
- `onSettingsChange`: event subscription.
- `getModelOptions`: available model list.
- `getDefaultSettings`: default settings object.

#### core/summarizer.js

- `formatSummary`: applies bullets or TL;DR.
- `summarizeText`: sends `/api/summarize` request and parses response.

#### chrome_extension/popup.js

- `setStatus`, `setResult`: UI helpers for popup state.
- `extractPageData`: collects title/url/body text from active tab.
- `populateModels`: fills model dropdown from settings.
- `summarize`: full popup flow from text extraction to API call.

#### chrome_extension/core/*

- `modelRouter.js`: same routing rules as web UI.
- `settingsManager.js`: local storage persistence for extension.
- `summarizer.js`: API wrapper for extension.

#### static/app.js

- Handles UI interactions, model selection payloads, and animation.
- Calls `/api/summarize` or `/api/summarize-upload` and renders results.

#### static/history.js

- Fetches `/api/history` and renders summary history.

#### static/settings.js

- Loads settings, binds form fields, and persists updates.

#### templates/index.html

- UI markup for URL/file input, model selection, and results.

#### templates/settings.html

- UI markup for settings controls.

#### templates/history.html

- UI markup for summary history.

### Rules and Constraints

- Model fallback rules ensure a summary is returned if the primary fails.
- Input length is capped by settings and chunking is applied for long texts.
- TF-IDF is always available as the fallback and as the first stage of TF-IDF + PGN.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Visualization Metrics and Comparisons

The visualization folder generates terminal-only comparison charts that summarize model quality, latency, energy, and size-specific behavior.

### Core Metrics (Calculated in `visualization/collect_metrics.py`)

- Quality overall:
  - `quality_overall = 100 * (0.5*ROUGE-L + 0.3*ROUGE-2 + 0.2*ROUGE-1) * (1 - 0.3*avg_repetition_rate)`
- Efficiency score:
  - `efficiency_score = 100 * (0.5*norm(ROUGE-L) + 0.3*norm(1/latency) + 0.2*norm(1/energy))`
- Latency, throughput, and compression:
  - `avg_latency_ms`, `avg_latency_per_input_word_ms`, `throughput_input_words_per_sec`, `compression_ratio`
- Energy and carbon:
  - `power_watts`, `energy_kwh`, `carbon_kg`
- Reliability and summary quality:
  - `error_rate`, `success_rate`, `avg_unique_word_ratio`, `avg_repetition_rate`

### Key Visualizations (Important Outputs)

- Quality, efficiency, and energy comparisons:
  - `visualization/outputs/compare_quality_overall.png`
  - `visualization/outputs/compare_efficiency_score.png`
  - `visualization/outputs/compare_energy_kwh.png`
  - `visualization/outputs/compare_carbon_kg.png`
- Model behavior by input size:
  - `visualization/outputs/scenario_quality_by_size.png`
  - `visualization/outputs/scenario_latency_by_size.png`
- Quality trade-offs:
  - `visualization/outputs/scatter_quality_latency.png`
  - `visualization/outputs/scatter_quality_energy.png`

### Summary of Observations (Captured in Scenario Charts)

- BART-Large is strongest on large inputs.
- PEGASUS is best on short/medium/large overall.
- T5-3B excels on huge inputs but has the highest latency.
- T5-Small underperforms on quality.
- TF-IDF / TF-IDF+PGN are fastest but lower quality than pretrained transformers.

### How to Run

```sh
python visualization/collect_metrics.py --samples 3 --transformers-mode offline
python visualization/plot_metric_comparisons.py
python visualization/plot_additional_comparisons.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Oxlac/AI-News-Summariser.git
   ```
2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app
   ```sh
   python app.py
   ```

### Optional: Train TF-IDF Vectorizer

```sh
python -m ml.train --config ml/configs/cnn_dm_bart.json
```

### Optional: Train PGN

```sh
python -m ml.pgn --train
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

- PGN training is computationally heavier than extractive TF-IDF and needs sufficient data/epochs.
- The TF-IDF baseline prioritizes frequency-weighted sentences, which can omit discourse-level coherence.
- Heuristic entity extraction and key-phrase mining are lightweight and not NER-grade.
- Article parsing depends on the structure of the source site; some pages may fail.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome. Ensure that code passes the ruff linter before submitting a PR.

## License

Distributed under the MIT License. See LICENSE.txt for more information.

## Contact

Your Name - [@Oxlac_](https://twitter.com/Oxlac_) - contact@oxlac.com

Discord Server - [https://discord.gg/2YdnSGHdET](https://discord.gg/2YdnSGHdET)

Project Link: [https://github.com/Oxlac/AI-News-Summariser](https://github.com/oxlac/mr.dm)

Developer: [Aadityaa Nagarajan](https://aadinagarajan.com)
