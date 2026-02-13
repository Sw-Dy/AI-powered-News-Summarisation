# Title of the Project: Advanced Multimodal Text Summarization System with Web Interface
**Mentor:** Dr. Santanu Roy  
**Duration:** 10 Weeks  

## 0) Project Goal (From Scratch)
Build an end-to-end system that:
- Ingests news articles from RSS feeds and web pages (scraping + extraction).
- Stores raw articles, cleaned text, and generated summaries in a database and an on-disk cache.
- Supports multiple summarization backends:
  - Baseline extractive (TF‑IDF + sentence ranking)
  - Stronger extractive (existing repo model / lightweight ML)
  - Abstractive (PGN and/or BART-family models, optional fine-tuning)
- Evaluates outputs using ROUGE (and optionally BERTScore) and tracks results per model/config.
- Exposes a web UI (and optional browser extension) for:
  - URL summarization
  - Batch ingestion + daily/weekly digests
  - History browsing, filters, exports (TXT/JSON/CSV)
  - Settings (length/style, model selection, performance limits)
- Adds reliability/security hardening: retries, rate limiting, sanitized inputs, user-agent policies.

## 1) Reference Architecture (High-Level)
**Backend**
- Web API: Flask (or FastAPI) with endpoints for ingest, summarize, history, export, health.
- Pipeline modules:
  - `ingestion/`: RSS fetcher, HTML downloader, readability/article parser
  - `preprocess/`: cleanup, dedup, language detection, sentence segmentation
  - `summarization/`:
    - `extractive_tfidf.py` (baseline)
    - `extractive_model.py` (repo’s model wrapper)
    - `abstractive_pgn.py` (PGN training + inference)
    - `abstractive_bart.py` (BART fine-tune + inference)
  - `evaluation/`: ROUGE, BERTScore, latency/cost tracking
  - `storage/`: DB + file cache + IDs
- Async/batch: scheduler (APScheduler/Celery/RQ) for periodic digests and batch runs.

**Database**
- Recommended: SQLite for local development; PostgreSQL for deployment.
- ORM: SQLAlchemy.
- Tables (minimum):
  - `sources` (rss_url, name, enabled)
  - `articles` (id, url, source_id, title, authors, publish_date, raw_html_path, cleaned_text, fetched_at, hash, language)
  - `summaries` (id, article_id, model_name, model_version, params_json, summary_text, created_at, rouge_json, latency_ms)
  - `digests` (id, period, created_at, config_json)
  - `digest_items` (digest_id, article_id, cluster_id, summary_id)

**File Cache**
- For reproducibility and offline re-runs:
  - `cache/raw_html/{article_id}.html`
  - `cache/clean_text/{article_id}.txt`
  - `cache/summaries/{summary_id}.json`

**Multimodal Extension (Lightweight)**
- Treat “multimodal” as: text + article metadata + lead image (optional).
- Minimum: store `top_image` URL, download thumbnails, show in UI.
- Optional: OCR (if images contain text) and include OCR text as extra input for summarization.

## 2) Core Functions / Modules To Create (Checklist)
**Ingestion**
- `fetch_rss_feed(feed_url) -> list[item]`
- `extract_article_urls(feed_items) -> list[url]`
- `download_url(url, timeout, headers, retries) -> html`
- `parse_article(html, url) -> {title, authors, publish_date, text, top_image}`
- `deduplicate_article(text, url) -> {hash, is_duplicate}`

**Preprocessing**
- `clean_text(text) -> text`
- `truncate_input(text, max_chars | max_tokens) -> text`
- `segment_sentences(text) -> list[sentence]`
- `normalize_whitespace(text) -> text`

**Summarization**
- Baseline TF‑IDF extractive:
  - `fit_tfidf_vectorizer(corpus) -> vectorizer`
  - `rank_sentences_tfidf(sentences, vectorizer) -> scores`
  - `summarize_extractive_tfidf(text, max_sentences) -> summary`
- Existing repo model wrapper:
  - `load_extractive_model(path) -> (vectorizer, model_config)`
  - `summarize_extractive(text, vectorizer, max_sentences) -> {summary, meta}`
- Abstractive PGN:
  - `train_pgn(dataset, config) -> checkpoints`
  - `pgn_infer(text, checkpoint, decoding_config) -> summary`
- Abstractive BART (or PEGASUS/T5):
  - `fine_tune_bart(dataset, config) -> checkpoints`
  - `bart_infer(text, checkpoint, decoding_config) -> summary`

**Evaluation**
- `compute_rouge(summary, reference) -> rouge_scores`
- `compute_bertscore(summary, reference) -> bertscore`
- `benchmark_latency(model_fn, inputs) -> stats`
- `log_eval_run(model_name, params, scores, latency)`

**Storage**
- `generate_article_id(url, publish_date, hash) -> id`
- `save_article(db, article, paths) -> article_id`
- `save_summary(db, article_id, summary, model_meta, eval_meta) -> summary_id`
- `load_cached_article(article_id) -> data`

**API + UI**
- `/api/ingest` (RSS URLs, source management)
- `/api/summarize` (URL or text)
- `/api/history` (filters, pagination)
- `/api/export` (TXT/JSON/CSV)
- `/api/digest/run` (batch)
- `/api/health` (status)

## 3) Training & Evaluation Plan (Models + Scores)
**Dataset**
- Start: small curated set (50–200 articles) with “reference” summaries:
  - Public datasets: CNN/DailyMail, XSum for training baseline abstractive models.
  - Project-specific: manually create references for a small set for demo-quality evaluation.

**Training**
1. TF‑IDF Extractive Baseline
   - No deep training; build vectorizer on corpus and rank sentences.
2. PGN (Pointer-Generator Network)
   - Train on CNN/DailyMail-style dataset.
   - Track: loss curves, validation ROUGE, length distribution, OOV/copy ratios.
3. BART Fine-tune (or other Transformer)
   - Fine-tune using Hugging Face Transformers.
   - Track: ROUGE-1/2/L, training time, GPU/CPU requirements.

**Evaluation Metrics**
- ROUGE-1, ROUGE-2, ROUGE-L (main automatic metric).
- Optional: BERTScore for semantic similarity.
- Operational metrics:
  - Latency (ms) per summary
  - Memory/CPU usage
  - Failure rate / retry counts

## 4) Full 10-Week Plan (Role-wise, With Technical Deliverables)

### Week 1
**Swagnik**
- Audit repository structure; list current pipeline capabilities and gaps.
- Document external review feedback and convert into actionable items (bugs, UX gaps, reliability gaps).

**Adrita**
- Gather requirements and define measurable success metrics:
  - Example: ROUGE-L target, max latency budget, ingestion success %, UI completion criteria.

**Triparno**
- Research summarization approaches (extractive vs abstractive) aligned to constraints:
  - CPU-only vs GPU
  - Offline capability vs hosted inference
  - Dependencies and runtime tradeoffs

**Trisha**
- Set up project management (issues, milestones, weekly deliverables).
- Draft system architecture diagram (ingest → preprocess → summarize → evaluate → store → UI).

### Week 2
**Swagnik**
- Implement/verify ingestion pipeline:
  - RSS fetcher + URL extraction
  - Robust HTML download (timeouts, retries, user-agent)
  - Article parsing (newspaper3k/readability)
- Add dedup (hashing), and basic source registry.

**Adrita**
- Create dataset plan:
  - Curated sample articles set
  - Reference summary annotation checklist
  - Evaluation checklist (coverage, factuality, length control)

**Triparno**
- Baseline summarizer integrated end-to-end:
  - TF‑IDF sentence ranking + configurable max sentences
  - Quick latency benchmark + basic ROUGE run on small set

**Trisha**
- Implement web app flow:
  - URL input → summarize → result card (title/authors/date/image/summary)
  - Clear error states and loading indicators

### Week 3
**Swagnik**
- Add stronger summarizer options:
  - Integrate repo’s existing extractive model wrapper
  - Compare against TF‑IDF baseline on sample set

**Adrita**
- Topic/category tagging plan:
  - Keyword-based taxonomy (economy, sports, politics, tech)
  - Optional lightweight model approach
- Define user-facing filters (by category/source/date).

**Triparno**
- Improve preprocessing:
  - Sentence segmentation and cleaning improvements
  - Boilerplate removal + length truncation strategy

**Trisha**
- Build initial UI to run full pipeline end-to-end:
  - Batch view of ingested articles
  - Summary panel with settings controls

### Week 4
**Swagnik**
- Add file-based cache + consistent IDs:
  - Save raw HTML, cleaned text, and summary outputs
  - Support re-run summaries without re-fetching

**Adrita**
- Refine topic/category tagging approach and finalize filters:
  - Validate tagging quality on sample set
  - Add source-level and category-level filter definitions

**Triparno**
- Add controllable summary parameters:
  - Length: short/medium/long (sentence-based or token-based)
  - Style: paragraph/bullets/TL;DR (formatting layer)
  - If LLM-based: prompt templates/config flags

**Trisha**
- Add export options + basic search/filter UI:
  - Export summary results as TXT/JSON/CSV
  - Search by title/keywords; filter by category/source/date

### Week 5
**Swagnik**
- Add scheduling and batch processing:
  - Daily/weekly digest job
  - Performance tuning for ingestion + summarization

**Adrita**
- Evaluate summarization quality:
  - Run ROUGE for baseline vs stronger models
  - Collect mentor/peer feedback; log issues and improvements

**Triparno**
- Polish UI behavior:
  - Loading states for ingestion/summarization
  - Clear error messages (invalid URL, fetch failed, parse failed)
  - Settings panel improvements

**Trisha**
- Add model fallback and factuality checks (lightweight):
  - Fallback order (extractive → abstractive)
  - Heuristics: avoid hallucinations by limiting to extractive when confidence low

### Week 6
**Swagnik**
- Add configuration system:
  - `.env` variables for model paths, DB URI, cache paths, rate limits
  - Environment profiles (dev/demo/prod)

**Adrita**
- Add unit tests for preprocessing/ingestion and integration test plan:
  - Parser tests with saved HTML fixtures
  - Dedup and cleaning tests

**Triparno**
- Add “daily digest” multi-article summarization:
  - Cluster articles by topic (keyword or embeddings)
  - Summarize cluster into a digest summary + per-article short summary

**Trisha**
- Add history page + basic analytics:
  - History list (past summaries)
  - Counts per category/source

### Week 7
**Swagnik**
- Security and reliability pass:
  - Rate limiting per IP
  - Retry/backoff for fetch
  - Sanitize inputs and restrict outbound requests to safe schemes

**Adrita**
- Run test suite and fix edge cases; document known limitations/risks:
  - Paywalls, JS-heavy sites, parsing failures
  - Language coverage limits

**Triparno**
- Optimize summarization latency/cost:
  - Caching + batching + max input length policies
  - CPU-friendly defaults; optional GPU acceleration if available

**Trisha**
- Model behavior analysis and end-to-end integration polishing:
  - Summary consistency checks (length, tone)
  - UI shows which model produced the summary and why

### Week 8
**Swagnik**
- Deployment setup:
  - Local reproducible install
  - One target: Docker/VM/cloud (choose one)
  - Add DB migration strategy and persistent storage config

**Adrita**
- Create demo dataset and scripted demo scenarios:
  - “Good” and “edge case” URLs
  - Acceptance checklist for done-criteria

**Triparno**
- Finalize evaluation report:
  - Baseline vs improved models comparison
  - Choose final default model and settings

**Trisha**
- Coordinate final web integration and demo readiness:
  - UI aligns to final pipeline behavior
  - Export and history fully working

### Week 9
**Swagnik**
- Full integration testing; fix deployment/runtime issues:
  - End-to-end ingestion + summarize + store + export
  - Stabilize configs and environment steps

**Adrita**
- Final QA pass against requirements:
  - Verify outputs, exports, filters, failure modes
  - Confirm success metrics are met (or log gaps)

**Triparno**
- Final model/pipeline tuning:
  - Quality improvements (preprocess + decoding params)
  - Ensure consistent results across categories

**Trisha**
- Finalize system documentation + architecture diagrams:
  - UI and API endpoints documented
  - Demo preparation support

### Week 10
**Swagnik**
- Compile final report sections: requirements, evaluation, limitations, future work.

**Adrita**
- Compile final report sections: requirements, evaluation, limitations, future work.

**Triparno**
- Compile final report sections: requirements, evaluation, limitations, future work.

**Trisha**
- Compile final report sections: requirements, evaluation, limitations, future work.

## 5) Team Sign-off Table (As Requested)
| Sl. No. | Name           | Class Roll No. | Signature |
|--------:|----------------|----------------|-----------|
| 1       | Swagnik Dey     | 2260021        |           |
| 2       | Adrita Sarkar   | 2260022        |           |
| 3       | Triparno Bose   | 2260051        |           |
| 4       | Trisha Pal      | 2260014        |           |

**Mentor’s Signature:**  

