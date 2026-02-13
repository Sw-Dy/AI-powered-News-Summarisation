import html as html_lib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

import nltk
import requests
import validators
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, make_response, send_from_directory
from newspaper import Article, Config
from textblob import TextBlob
from urllib.parse import urlparse
from werkzeug.utils import secure_filename

from ml.data_utils import normalize_text, chunk_text_by_tokens
from ml.pgn import SpTokenizer, PointerGenerator, build_extended_ids, beam_search, decode_tokens
from ml.summarizer import load_extractive_model, summarize_extractive
from services.video_summarizer import (
    AudioExtractionError,
    SummarizationError,
    TranscriptionFailedError,
    VideoProcessingError,
    VideoSummarizer,
)

nltk.download('punkt')

app = Flask(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

MODEL_DIR = os.getenv("MODEL_DIR", "outputs/cnn_dm_extractive")
TRAIN_CONFIG = os.getenv("TRAIN_CONFIG", "ml/configs/cnn_dm_bart.json")
AUTO_TRAIN = os.getenv("AUTO_TRAIN", "false").lower() == "true"
MAX_SUMMARY_SENTENCES = int(os.getenv("MAX_SUMMARY_SENTENCES", "5"))
DB_PATH = os.getenv("DB_PATH", os.path.join(app.root_path, "data", "app.db"))
DEFAULT_HEADERS = {
    "User-Agent": os.getenv(
        "HTTP_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
ALLOWED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv"
}

DEFAULT_SETTINGS = {
    "general": {
        "defaultModel": "auto",
        "fallbackModel": "tfidf-pgn",
        "summaryLength": "medium",
        "outputFormat": "paragraph",
        "languageStyle": "neutral"
    },
    "analysis": {
        "keyPhrases": True,
        "namedEntities": True,
        "sentiment": True,
        "readability": True,
        "questionAnswering": True
    },
    "performance": {
        "maxInputLength": 12000,
        "chunkSize": 2000,
        "streaming": False
    },
    "extension": {
        "compactMode": True,
        "shortSummaryOnly": True,
        "deepLink": True
    }
}

ABSTRACTIVE_MODEL_MAP = {
    "t5-small": "t5-small",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "bart-large": "facebook/bart-large-cnn",
    "pegasus": "google/pegasus-cnn_dailymail",
    "t5-3b": "t5-3b",
}

_abstractive_models = {}
_abstractive_tokenizers = {}
_abstractive_device = None
_pgn_model = None
_pgn_tokenizer = None
_pgn_device = None

def deep_merge(base, next_value):
    if not isinstance(next_value, dict):
        return base
    output = dict(base)
    for key, value in next_value.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            output[key] = deep_merge(base.get(key, {}), value)
        else:
            output[key] = value
    return output

def build_full_settings(settings_payload):
    if not isinstance(settings_payload, dict):
        return deep_merge(DEFAULT_SETTINGS, {})
    return deep_merge(DEFAULT_SETTINGS, settings_payload)

def resolve_settings(settings_payload):
    stored = load_settings_from_db() or {}
    merged = deep_merge(DEFAULT_SETTINGS, stored)
    if isinstance(settings_payload, dict) and settings_payload:
        return deep_merge(merged, settings_payload)
    return merged

def allowed_file(filename):
    _, ext = os.path.splitext(filename or "")
    return ext.lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, extension):
    if extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    if extension == ".pdf":
        try:
            import PyPDF2
        except ImportError as exc:
            raise RuntimeError("PyPDF2 is not installed") from exc
        with open(file_path, "rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            return "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
    if extension in {".png", ".jpg", ".jpeg"}:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr is not installed") from exc
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(file_path, detail=0)
        return "\n".join(results)
    return ""

def ensure_db_dir():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

def get_db_connection():
    ensure_db_dir()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    with get_db_connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                url TEXT,
                title TEXT,
                authors TEXT,
                publish_date TEXT,
                summary TEXT,
                sentiment TEXT,
                selected_model TEXT,
                input_text TEXT,
                settings_json TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data_json TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()

def load_settings_from_db():
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT data_json FROM settings WHERE id = 1"
        ).fetchone()
        if not row or not row["data_json"]:
            return None
        try:
            return json.loads(row["data_json"])
        except json.JSONDecodeError:
            return None

def save_settings_to_db(settings):
    payload = json.dumps(settings)
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO settings (id, data_json, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                data_json = excluded.data_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (payload,)
        )
        connection.commit()

def record_summary(
    url,
    title,
    authors,
    publish_date,
    summary,
    sentiment,
    selected_model,
    input_text,
    settings_payload
):
    settings_json = json.dumps(settings_payload) if settings_payload else None
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO summaries (
                url,
                title,
                authors,
                publish_date,
                summary,
                sentiment,
                selected_model,
                input_text,
                settings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                title,
                authors,
                publish_date,
                summary,
                sentiment,
                selected_model,
                input_text,
                settings_json
            )
        )
        connection.commit()

def fetch_history(limit=25):
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                created_at,
                url,
                title,
                authors,
                publish_date,
                summary,
                sentiment,
                selected_model,
                settings_json
            FROM summaries
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            settings_raw = item.pop("settings_json", None)
            if settings_raw:
                try:
                    item["settings"] = json.loads(settings_raw)
                except json.JSONDecodeError:
                    item["settings"] = None
            else:
                item["settings"] = None
            items.append(item)
        return items

def model_is_ready(path):
    return os.path.exists(os.path.join(path, "model.pkl"))


def train_model():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.train",
            "--config",
            TRAIN_CONFIG,
        ],
        check=True,
    )


init_db()
if not load_settings_from_db():
    save_settings_to_db(DEFAULT_SETTINGS)

if AUTO_TRAIN and not model_is_ready(MODEL_DIR):
    train_model()

if not model_is_ready(MODEL_DIR):
    raise RuntimeError(f"Model not found at {MODEL_DIR}. Train it first.")

vectorizer, model_config = load_extractive_model(MODEL_DIR)
configured_max_sentences = int(model_config.get("max_sentences", MAX_SUMMARY_SENTENCES))

def get_website_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def normalize_url(raw_url):
    if not raw_url:
        return raw_url
    parsed = urlparse(raw_url)
    if parsed.scheme:
        return raw_url
    return f"https://{raw_url}"


def fetch_url(url):
    session = requests.Session()
    return session.get(url, headers=DEFAULT_HEADERS, timeout=30, allow_redirects=True)


def cors_json(payload, status=200):
    response = make_response(jsonify(payload), status)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def summary_length_to_sentences(length_value):
    base = configured_max_sentences
    if length_value == "short":
        return max(2, base - 2)
    if length_value == "long":
        return base + 3
    return base

def sentence_split(text):
    if not text:
        return []
    try:
        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

def chunk_text_by_sentences(text, chunk_size):
    if not text:
        return []
    if not chunk_size or chunk_size <= 0:
        return [text]
    sentences = sentence_split(text)
    if not sentences:
        return [text]
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]

def summarize_extractive_with_chunks(text, max_sentences, chunk_size):
    if not text:
        return ""
    if not chunk_size or len(text) <= chunk_size:
        result = summarize_extractive(text=text, vectorizer=vectorizer, max_sentences=max_sentences)
        return result.get("summary", "").strip()
    chunks = chunk_text_by_sentences(text, chunk_size)
    if not chunks:
        return ""
    per_chunk = max(1, round(max_sentences / max(len(chunks), 1)))
    summaries = []
    for chunk in chunks:
        result = summarize_extractive(text=chunk, vectorizer=vectorizer, max_sentences=per_chunk)
        chunk_summary = result.get("summary", "").strip()
        if chunk_summary:
            summaries.append(chunk_summary)
    if not summaries:
        return ""
    combined = " ".join(summaries)
    sentences = sentence_split(combined)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    return " ".join(sentences)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "while", "to", "of", "in", "on", "for",
    "with", "as", "by", "at", "from", "into", "over", "under", "between", "after", "before", "about", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those", "we", "they", "he", "she",
    "you", "i", "me", "my", "our", "their", "them", "his", "her", "not", "no", "yes", "do", "does", "did", "so",
}

def extract_key_phrases(text, limit=8):
    words = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
    if not words:
        return []
    counts = {}
    for word in words:
        if word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    if not counts:
        return []
    sorted_words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_words = {word for word, _ in sorted_words[: max(limit * 2, limit)]}
    phrases = []
    for i in range(len(words) - 1):
        first = words[i]
        second = words[i + 1]
        if first in _STOPWORDS or second in _STOPWORDS:
            continue
        if first in top_words and second in top_words:
            phrase = f"{first} {second}"
            if phrase not in phrases:
                phrases.append(phrase)
        if len(phrases) >= limit:
            break
    if not phrases:
        phrases = [word for word, _ in sorted_words[:limit]]
    return phrases[:limit]

def extract_named_entities(text, limit=10):
    candidates = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    if not candidates:
        return []
    seen = set()
    entities = []
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        if normalized.lower() in _STOPWORDS:
            continue
        entities.append(normalized)
        seen.add(normalized)
        if len(entities) >= limit:
            break
    return entities

def count_syllables(word):
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in cleaned:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def compute_readability(text):
    sentences = sentence_split(text)
    words = re.findall(r"[a-zA-Z]+", text)
    if not sentences or not words:
        return None
    syllables = sum(count_syllables(word) for word in words)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllables / word_count)
    return round(score, 2)

def apply_language_style(summary, style):
    if not summary:
        return summary
    if style == "simple":
        sentences = sentence_split(summary)
        simplified = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 24:
                simplified.append(" ".join(words[:24]).rstrip(".") + ".")
            else:
                simplified.append(sentence)
        return " ".join(simplified)
    return summary

def apply_output_format(summary, output_format):
    if not summary:
        return summary
    if output_format == "bullets":
        sentences = [sentence.strip() for sentence in summary.split(". ") if sentence.strip()]
        return "\n".join([f"• {sentence.rstrip('.')}" for sentence in sentences])
    if output_format == "tldr":
        first_sentence = [sentence.strip() for sentence in summary.split(". ") if sentence.strip()]
        first = first_sentence[0] if first_sentence else summary
        return f"TL;DR: {first.rstrip('.')}."
    return summary

def build_qa_pairs(summary):
    if not summary:
        return []
    sentences = sentence_split(summary)
    first = sentences[0] if sentences else summary
    return [{"question": "What is this article about?", "answer": first}]

def select_model_id(settings_payload):
    selected = settings_payload.get("selectedModel") or settings_payload.get("general", {}).get("defaultModel", "")
    if not selected or selected == "auto":
        return settings_payload.get("general", {}).get("fallbackModel", "tfidf-pgn")
    return selected

def get_fallback_model(settings_payload):
    return settings_payload.get("general", {}).get("fallbackModel", "tfidf-pgn")

def get_abstractive(model_id):
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError as exc:
        raise RuntimeError("transformers is not installed") from exc
    global _abstractive_device
    if _abstractive_device is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed") from exc
        _abstractive_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = ABSTRACTIVE_MODEL_MAP.get(model_id, model_id)
    if model_name not in _abstractive_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_abstractive_device)
        _abstractive_models[model_name] = model
        _abstractive_tokenizers[model_name] = tokenizer
    return _abstractive_tokenizers[model_name], _abstractive_models[model_name], _abstractive_device, model_name

def summarize_abstractive(text, model_id):
    tokenizer, model, device, model_name = get_abstractive(model_id)
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=512, stride=64)
    summaries = []
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed") from exc
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output_ids = model.generate(
                **inputs,
                num_beams=4,
                max_length=128,
                min_length=24,
                do_sample=False,
            )
            summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return normalize_text(" ".join(summaries)), model_name

def pgn_args():
    class Args:
        pass
    args = Args()
    args.pgn_checkpoint = os.getenv("PGN_CHECKPOINT", "outputs/pgn_smoke/checkpoint_epoch_1.pt")
    args.pgn_tokenizer = os.getenv("PGN_TOKENIZER", "")
    args.beam_size = int(os.getenv("PGN_BEAM_SIZE", "4"))
    args.max_summary_len = int(os.getenv("PGN_MAX_SUMMARY_LEN", "100"))
    args.coverage_penalty = float(os.getenv("PGN_COVERAGE_PENALTY", "1.0"))
    args.length_penalty = float(os.getenv("PGN_LENGTH_PENALTY", "1.2"))
    args.no_repeat_ngram_size = int(os.getenv("PGN_NO_REPEAT_NGRAM_SIZE", "3"))
    args.max_source_len = int(os.getenv("PGN_MAX_SOURCE_LEN", "400"))
    args.hidden_size = int(os.getenv("PGN_HIDDEN_SIZE", "128"))
    args.embed_size = int(os.getenv("PGN_EMBED_SIZE", "128"))
    args.dropout = float(os.getenv("PGN_DROPOUT", "0.3"))
    return args

def get_pgn():
    global _pgn_model, _pgn_tokenizer, _pgn_device
    if _pgn_model and _pgn_tokenizer and _pgn_device:
        return _pgn_model, _pgn_tokenizer, _pgn_device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed") from exc
    args = pgn_args()
    checkpoint = torch.load(args.pgn_checkpoint, map_location="cpu")
    tokenizer_path = args.pgn_tokenizer or checkpoint.get("tokenizer_path", "")
    if not tokenizer_path:
        raise RuntimeError("tokenizer_path_missing")
    _pgn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SpTokenizer(tokenizer_path)
    model = PointerGenerator(tokenizer.vocab_size, args.embed_size, args.hidden_size, args.dropout).to(_pgn_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _pgn_model = model
    _pgn_tokenizer = tokenizer
    return _pgn_model, _pgn_tokenizer, _pgn_device

def summarize_with_pgn(text):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed") from exc
    args = pgn_args()
    model, tokenizer, device = get_pgn()
    source_ids, source_pieces = tokenizer.encode_pieces(
        text, add_bos=False, add_eos=True, max_len=args.max_source_len
    )
    source_ext_ids, oovs = build_extended_ids(tokenizer, source_ids, source_pieces)
    batch = {
        "source_ids": torch.tensor([source_ids], dtype=torch.long),
        "source_ext_ids": torch.tensor([source_ext_ids], dtype=torch.long),
        "source_lens": torch.tensor([len(source_ids)], dtype=torch.long),
        "max_oov": len(oovs),
        "oovs": [oovs],
    }
    tokens = beam_search(
        model,
        tokenizer,
        batch,
        device,
        args.beam_size,
        args.max_summary_len,
        args.coverage_penalty,
        args.length_penalty,
        args.no_repeat_ngram_size,
    )
    return decode_tokens(tokenizer, tokens, oovs)

def is_gibberish(summary_text):
    words = re.findall(r"[a-zA-Z0-9]+", summary_text.lower())
    if len(words) < 8:
        return True
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.35:
        return True
    run = 1
    best_run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 1
    return best_run >= 6

def summarize_with_model(text, selected_model, fallback_model, max_sentences, chunk_size):
    primary = selected_model
    fallback = fallback_model if fallback_model and fallback_model != primary else ""
    summary = ""
    model_used = primary or "tfidf"
    try:
        if primary == "tfidf-pgn":
            tfidf_text = summarize_extractive_with_chunks(text, max_sentences, chunk_size)
            if not tfidf_text:
                raise RuntimeError("tfidf_empty")
            pgn_summary = summarize_with_pgn(tfidf_text)
            if not pgn_summary or is_gibberish(pgn_summary):
                summary = tfidf_text
                model_used = "tfidf"
            else:
                summary = pgn_summary
                model_used = "tfidf-pgn"
        elif primary in ABSTRACTIVE_MODEL_MAP:
            summary, model_name = summarize_abstractive(text, primary)
            model_used = model_name
        elif primary in {"tfidf", "extractive"}:
            summary = summarize_extractive_with_chunks(text, max_sentences, chunk_size)
            model_used = "tfidf"
        else:
            summary = summarize_extractive_with_chunks(text, max_sentences, chunk_size)
            model_used = "tfidf"
    except Exception as exc:
        app.logger.error("Primary model failed (%s): %s", primary, exc)
        summary = ""
    if not summary and fallback:
        summary, model_used = summarize_with_model(text, fallback, "", max_sentences, chunk_size)
    return summary, model_used


def parse_article(url, html_content=None):
    config = Config()
    config.browser_user_agent = DEFAULT_HEADERS["User-Agent"]
    config.request_timeout = 30
    article = Article(url, config=config)
    if html_content:
        article.download(input_html=html_content)
    else:
        article.download()
    article.parse()
    article.nlp()
    title = article.title
    authors = ", ".join(article.authors)
    if not authors:
        authors = get_website_name(url)
    publish_date = article.publish_date.strftime('%B %d, %Y') if article.publish_date else "N/A"
    top_image = article.top_image
    return article.text, title, authors, publish_date, top_image


def extract_title_from_html(html_content):
    if not html_content:
        return ""
    match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return html_lib.unescape(title)


def extract_text_from_html(html_content):
    if not html_content:
        return ""
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html_content)
    cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
    cleaned = html_lib.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "happy 😊"
    if polarity < 0:
        return " sad 😟"
    return "neutral 😐"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = normalize_url(request.form['url'])
        if not validators.url(url):
            flash('Please enter a valid URL.')
            return redirect(url_for('index'))
        
        response = None
        try:
            response = fetch_url(url)
        except requests.RequestException:
            response = None
        html_body = response.text if response and response.text else ""
        try:
            article_text, title, authors, publish_date, top_image = parse_article(url, html_body or None)
        except Exception:
            article_text = extract_text_from_html(html_body)
            title = title or extract_title_from_html(html_body)
            authors = authors or get_website_name(url)
            publish_date = publish_date or "N/A"
            top_image = top_image or ""
        if not article_text:
            flash('Failed to download the content of the URL.')
            return redirect(url_for('index'))
        result = summarize_extractive(
            text=article_text,
            vectorizer=vectorizer,
            max_sentences=configured_max_sentences,
        )
        summary = result.get("summary", "")
        if summary == "":
            flash('Please enter a valid URL.')
            return redirect(url_for('index'))
        sentiment = analyze_sentiment(article_text)
        record_summary(
            url=url,
            title=title,
            authors=authors,
            publish_date=publish_date,
            summary=summary,
            sentiment=sentiment,
            selected_model="extractive",
            input_text=article_text,
            settings_payload=build_full_settings(
                {
                    "general": {
                        "defaultModel": "extractive"
                    },
                    "performance": {
                        "maxInputLength": len(article_text)
                    },
                    "selectedModel": "extractive"
                }
            )
        )

        return render_template('index.html', title=title, authors=authors, publish_date=publish_date, summary=summary, top_image=top_image, sentiment=sentiment)

    return render_template('index.html')


@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/core/<path:filename>")
def core_assets(filename):
    core_dir = os.path.join(app.root_path, "core")
    return send_from_directory(core_dir, filename)


@app.route("/api/summarize", methods=["POST", "OPTIONS"])
def api_summarize():
    if request.method == "OPTIONS":
        return cors_json({})
    try:
        payload = request.get_json(silent=True) or {}
        text = (payload.get("text") or "").strip()
        url = normalize_url((payload.get("url") or "").strip())
        title = (payload.get("title") or "").strip()
        settings_payload = payload.get("settings")
        full_settings = resolve_settings(settings_payload)
        summary_length = full_settings.get("general", {}).get("summaryLength", "medium")
        output_format = full_settings.get("general", {}).get("outputFormat", "paragraph")
        language_style = full_settings.get("general", {}).get("languageStyle", "neutral")
        max_sentences = summary_length_to_sentences(summary_length)
        max_input_length = int(full_settings.get("performance", {}).get("maxInputLength", 12000))
        chunk_size = int(full_settings.get("performance", {}).get("chunkSize", 2000))
        streaming_enabled = bool(full_settings.get("performance", {}).get("streaming", False))
        authors = ""
        publish_date = ""
        top_image = ""
        sentiment = ""
        if not text and url:
            if not validators.url(url):
                return cors_json({"error": "invalid_url"}, status=400)
            fetch_error = None
            response = None
            try:
                response = fetch_url(url)
            except requests.RequestException as exc:
                fetch_error = exc
            parse_error = None
            try:
                html_body = response.text if response and response.text else ""
                text, article_title, authors, publish_date, top_image = parse_article(
                    url,
                    html_body or None
                )
            except Exception as exc:
                parse_error = exc
                text = ""
            if not text and response and response.text:
                text = extract_text_from_html(response.text)
                if not title:
                    title = extract_title_from_html(response.text)
            if not text:
                if fetch_error and not response:
                    return cors_json({"error": "fetch_failed", "message": str(fetch_error)}, status=400)
                if parse_error:
                    return cors_json({"error": "parse_failed", "message": str(parse_error)}, status=400)
                return cors_json({"error": "fetch_failed", "message": "no_readable_text"}, status=400)
            if not title:
                title = article_title
        if not text:
            if url:
                return cors_json({"error": "fetch_failed", "message": "no_readable_text"}, status=400)
            return cors_json({"error": "empty_text"}, status=400)
        if len(text) > max_input_length:
            text = text[:max_input_length]
        selected_model = select_model_id(full_settings)
        fallback_model = get_fallback_model(full_settings)
        summary, model_used = summarize_with_model(text, selected_model, fallback_model, max_sentences, chunk_size)
        if not summary:
            return cors_json({"error": "summary_failed"}, status=500)
        if full_settings.get("analysis", {}).get("sentiment", True):
            sentiment = analyze_sentiment(text)
        selected_model = model_used
        formatted_summary = apply_output_format(apply_language_style(summary, language_style), output_format)
        analysis_settings = full_settings.get("analysis", {})
        analysis_payload = {}
        if analysis_settings.get("keyPhrases"):
            analysis_payload["key_phrases"] = extract_key_phrases(text)
        if analysis_settings.get("namedEntities"):
            analysis_payload["named_entities"] = extract_named_entities(text)
        if analysis_settings.get("readability"):
            analysis_payload["readability"] = compute_readability(text)
        if analysis_settings.get("questionAnswering"):
            analysis_payload["qa_pairs"] = build_qa_pairs(summary)
        response_payload = {
            "summary": summary,
            "formatted_summary": formatted_summary,
            "title": title,
            "authors": authors,
            "publish_date": publish_date,
            "top_image": top_image,
            "sentiment": sentiment,
            "selected_model": selected_model,
            "analysis": analysis_payload,
            "output_format": output_format,
            "language_style": language_style,
            "streaming_enabled": streaming_enabled,
        }
        record_summary(
            url=url,
            title=title,
            authors=authors,
            publish_date=publish_date,
            summary=summary,
            sentiment=sentiment,
            selected_model=selected_model,
            input_text=text,
            settings_payload=full_settings
        )
        return cors_json(response_payload)
    except Exception as exc:
        app.logger.exception("api_summarize failed")
        return cors_json({"error": "server_error", "message": str(exc)}, status=500)


@app.route("/api/summarize-upload", methods=["POST", "OPTIONS"])
def api_summarize_upload():
    if request.method == "OPTIONS":
        return cors_json({})
    try:
        if "file" not in request.files:
            return cors_json({"error": "missing_file"}, status=400)
        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename or "")
        if not filename:
            return cors_json({"error": "invalid_file"}, status=400)
        if not allowed_file(filename):
            return cors_json({"error": "unsupported_file"}, status=400)
        settings_payload = None
        raw_settings = request.form.get("settings")
        if raw_settings:
            try:
                settings_payload = json.loads(raw_settings)
            except json.JSONDecodeError:
                settings_payload = None
        full_settings = resolve_settings(settings_payload)
        summary_length = full_settings.get("general", {}).get("summaryLength", "medium")
        output_format = full_settings.get("general", {}).get("outputFormat", "paragraph")
        language_style = full_settings.get("general", {}).get("languageStyle", "neutral")
        max_sentences = summary_length_to_sentences(summary_length)
        max_input_length = int(full_settings.get("performance", {}).get("maxInputLength", 12000))
        chunk_size = int(full_settings.get("performance", {}).get("chunkSize", 2000))
        streaming_enabled = bool(full_settings.get("performance", {}).get("streaming", False))
        selected_model = select_model_id(full_settings)
        fallback_model = get_fallback_model(full_settings)
        title = request.form.get("title") or filename
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, filename)
            uploaded.save(file_path)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".avi", ".mkv"}:
                video_summarizer = VideoSummarizer(max_sentences=max_sentences)
                video_result = video_summarizer.transcribe_video(file_path)
                text = video_result.get("transcription", "")
                summary, model_used = summarize_with_model(text, selected_model, fallback_model, max_sentences, chunk_size)
                selected_model = model_used
            else:
                text = extract_text_from_file(file_path, ext)
                if not text:
                    return cors_json({"error": "empty_text"}, status=400)
                if len(text) > max_input_length:
                    text = text[:max_input_length]
                summary, model_used = summarize_with_model(text, selected_model, fallback_model, max_sentences, chunk_size)
                selected_model = model_used
        if not summary:
            return cors_json({"error": "summary_failed"}, status=500)
        if len(text) > max_input_length:
            text = text[:max_input_length]
        sentiment = ""
        if full_settings.get("analysis", {}).get("sentiment", True):
            sentiment = analyze_sentiment(text)
        formatted_summary = apply_output_format(apply_language_style(summary, language_style), output_format)
        analysis_settings = full_settings.get("analysis", {})
        analysis_payload = {}
        if analysis_settings.get("keyPhrases"):
            analysis_payload["key_phrases"] = extract_key_phrases(text)
        if analysis_settings.get("namedEntities"):
            analysis_payload["named_entities"] = extract_named_entities(text)
        if analysis_settings.get("readability"):
            analysis_payload["readability"] = compute_readability(text)
        if analysis_settings.get("questionAnswering"):
            analysis_payload["qa_pairs"] = build_qa_pairs(summary)
        response_payload = {
            "summary": summary,
            "formatted_summary": formatted_summary,
            "title": title,
            "authors": "",
            "publish_date": "",
            "top_image": "",
            "sentiment": sentiment,
            "selected_model": selected_model,
            "analysis": analysis_payload,
            "output_format": output_format,
            "language_style": language_style,
            "streaming_enabled": streaming_enabled,
        }
        record_summary(
            url="",
            title=title,
            authors="",
            publish_date="",
            summary=summary,
            sentiment=sentiment,
            selected_model=selected_model,
            input_text=text,
            settings_payload=full_settings
        )
        return cors_json(response_payload)
    except (AudioExtractionError, TranscriptionFailedError, SummarizationError, VideoProcessingError) as exc:
        app.logger.exception("api_summarize_upload failed")
        return cors_json({"error": "video_processing_failed", "message": str(exc)}, status=400)
    except Exception as exc:
        app.logger.exception("api_summarize_upload failed")
        return cors_json({"error": "server_error", "message": str(exc)}, status=500)


@app.route("/api/settings", methods=["GET", "POST", "OPTIONS"])
def api_settings():
    if request.method == "OPTIONS":
        return cors_json({})
    if request.method == "GET":
        stored = load_settings_from_db()
        settings_payload = deep_merge(DEFAULT_SETTINGS, stored or {})
        return cors_json({"settings": settings_payload})
    payload = request.get_json(silent=True) or {}
    settings_payload = payload.get("settings")
    if settings_payload is None:
        settings_payload = payload
    if not isinstance(settings_payload, dict):
        return cors_json({"error": "invalid_settings"}, status=400)
    merged_settings = deep_merge(DEFAULT_SETTINGS, settings_payload)
    save_settings_to_db(merged_settings)
    return cors_json({"settings": merged_settings})


@app.route("/api/history", methods=["GET", "OPTIONS"])
def api_history():
    if request.method == "OPTIONS":
        return cors_json({})
    try:
        limit = int(request.args.get("limit", "25"))
    except ValueError:
        limit = 25
    limit = max(1, min(limit, 100))
    items = fetch_history(limit)
    return cors_json({"items": items})

app.secret_key = 'your_secret_key'

if __name__ == '__main__':
    app.run(debug=True)
