import json
import os
import pickle
import time

from ml.data_utils import normalize_text


def load_extractive_model(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        vectorizer = pickle.load(f)
    config_path = os.path.join(model_dir, "model_config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return vectorizer, config


def summarize_extractive(text, vectorizer, max_sentences=5):
    start = time.time()
    text = normalize_text(text)
    if not text:
        return {"summary": "", "latency_ms": 0.0}
    try:
        from nltk.tokenize import sent_tokenize
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        words = text.split()
        if not words:
            return {"summary": "", "latency_ms": 0.0}
        chunk_size = max(8, min(30, len(words)))
        sentences = [
            " ".join(words[i : i + chunk_size]).strip()
            for i in range(0, len(words), chunk_size)
        ]
    if not sentences:
        return {"summary": "", "latency_ms": 0.0}
    if len(sentences) <= max_sentences:
        return {"summary": ". ".join(sentences), "latency_ms": (time.time() - start) * 1000.0}
    tfidf = vectorizer.transform(sentences)
    if tfidf.shape[1] == 0:
        summary = ". ".join(sentences[:max_sentences])
        return {"summary": summary, "latency_ms": (time.time() - start) * 1000.0}
    scores = tfidf.sum(axis=1).A1
    top_idx = scores.argsort()[-max_sentences:][::-1]
    selected = [sentences[i] for i in sorted(top_idx)]
    summary = ". ".join(selected)
    latency_ms = (time.time() - start) * 1000.0
    return {"summary": summary, "latency_ms": latency_ms}
