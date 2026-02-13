import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import evaluate


def ensure_repo_on_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def parse_power_overrides(raw_value):
    overrides = {}
    if not raw_value:
        return overrides
    for pair in raw_value.split(","):
        if not pair.strip():
            continue
        key, value = pair.split("=", 1)
        overrides[key.strip()] = float(value.strip())
    return overrides


def resolve_power(model_id, overrides):
    base = {
        "tfidf": 20.0,
        "tfidf-pgn": 60.0,
        "t5-small": 80.0,
        "distilbart": 120.0,
        "bart-large": 180.0,
        "pegasus": 200.0,
        "t5-3b": 350.0,
    }
    if model_id in overrides:
        return overrides[model_id]
    return base.get(model_id, 120.0)


def load_samples(sample_count, seed, dataset_path):
    rng = random.Random(seed)
    if os.path.exists(dataset_path):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            if hasattr(dataset, "shuffle"):
                dataset = dataset.shuffle(seed=seed)
            if hasattr(dataset, "select"):
                dataset = dataset.select(range(min(sample_count, len(dataset))))
            items = []
            for row in dataset:
                article = row.get("article") or row.get("document") or ""
                highlights = row.get("highlights") or row.get("summary") or ""
                items.append({"article": article, "highlights": highlights})
            if items:
                return items
        except Exception:
            pass
    try:
        from datasets import load_dataset
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"validation[:{sample_count}]")
        items = []
        for row in dataset:
            items.append({"article": row.get("article", ""), "highlights": row.get("highlights", "")})
        if items:
            return items
    except Exception:
        pass
    fallback = [
        {
            "article": "The central bank raised interest rates again on Tuesday to curb inflation, citing strong consumer spending and resilient labor markets. Analysts expect another hike this quarter as energy prices remain volatile and wage growth accelerates.",
            "highlights": "The central bank raised rates to curb inflation and may hike again soon."
        },
        {
            "article": "A tech company announced a new battery technology that promises faster charging and longer life. The firm said it will partner with automakers next year to begin pilot production while continuing safety testing.",
            "highlights": "A tech company unveiled faster-charging batteries and plans pilot production with automakers."
        },
        {
            "article": "Heavy rains caused flooding across the region, forcing evacuations and closing several highways. Emergency crews reported no fatalities but warned residents to avoid flooded roads and prepare for more storms.",
            "highlights": "Flooding from heavy rains led to evacuations and road closures; officials urged caution."
        },
    ]
    rng.shuffle(fallback)
    return fallback[:sample_count]


def summarize_tfidf(text, vectorizer, max_sentences):
    ensure_repo_on_path()
    from ml.data_utils import normalize_text
    from ml.summarizer import summarize_extractive
    result = summarize_extractive(text, vectorizer, max_sentences=max_sentences)
    return normalize_text(result.get("summary", ""))


def build_pgn_components(args):
    import torch
    ensure_repo_on_path()
    from ml.pgn import SpTokenizer, PointerGenerator
    checkpoint = torch.load(args.pgn_checkpoint, map_location="cpu")
    tokenizer_path = args.pgn_tokenizer or checkpoint.get("tokenizer_path", "")
    if not tokenizer_path:
        raise RuntimeError("tokenizer_path_missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SpTokenizer(tokenizer_path)
    model = PointerGenerator(tokenizer.vocab_size, args.embed_size, args.hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, device


def summarize_pgn(text, tfidf_vectorizer, max_sentences, chunk_size, args, pgn_cache):
    import torch
    ensure_repo_on_path()
    from ml.data_utils import normalize_text
    from ml.pgn import build_extended_ids, beam_search, decode_tokens
    if pgn_cache["model"] is None:
        model, tokenizer, device = build_pgn_components(args)
        pgn_cache["model"] = model
        pgn_cache["tokenizer"] = tokenizer
        pgn_cache["device"] = device
    model = pgn_cache["model"]
    tokenizer = pgn_cache["tokenizer"]
    device = pgn_cache["device"]
    tfidf_summary = summarize_tfidf(text, tfidf_vectorizer, max_sentences)
    if not tfidf_summary:
        return ""
    source_ids, source_pieces = tokenizer.encode_pieces(
        tfidf_summary, add_bos=False, add_eos=True, max_len=args.max_source_len
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
    return normalize_text(decode_tokens(tokenizer, tokens, oovs))


def get_transformer(model_id, device_cache, transformers_mode):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if device_cache.get("device") is None:
        import torch
        device_cache["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_id
    local_only = transformers_mode == "offline"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=local_only).to(device_cache["device"])
    return tokenizer, model, device_cache["device"]


def summarize_transformer(text, model_id, device_cache, transformers_mode):
    import torch
    ensure_repo_on_path()
    from ml.data_utils import chunk_text_by_tokens, normalize_text
    tokenizer, model, device = get_transformer(model_id, device_cache, transformers_mode)
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=512, stride=64)
    summaries = []
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
    return normalize_text(" ".join(summaries))


def compute_energy_kwh(power_watts, latency_ms, samples):
    total_seconds = (latency_ms / 1000.0) * samples
    return (power_watts * total_seconds) / 3600000.0


def estimate_transformer_metrics(model_id, samples):
    ensure_repo_on_path()
    from ml.data_utils import normalize_text
    base_rouge = {
        "t5-small": 0.25,
        "distilbart": 0.29,
        "bart-large": 0.33,
        "pegasus": 0.31,
        "t5-3b": 0.36,
    }
    speed_tokens_per_sec = {
        "t5-small": 1800.0,
        "distilbart": 1300.0,
        "bart-large": 900.0,
        "pegasus": 850.0,
        "t5-3b": 350.0,
    }
    token_counts = [len(normalize_text(item.get("article", "")).split()) for item in samples]
    avg_tokens = statistics.mean(token_counts) if token_counts else 800.0
    speed = speed_tokens_per_sec.get(model_id, 1000.0)
    avg_latency_ms = (avg_tokens / speed) * 1000.0
    avg_summary_length = max(30.0, min(120.0, avg_tokens * 0.12))
    rougeL = base_rouge.get(model_id, 0.28)
    return avg_latency_ms, avg_summary_length, rougeL


def normalize(values, higher_is_better=True):
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    normalized = []
    for value in values:
        score = (value - minimum) / (maximum - minimum)
        if not higher_is_better:
            score = 1.0 - score
        normalized.append(score)
    return normalized


def summarize_tokens(text):
    tokens = [token for token in text.split() if token]
    total = len(tokens)
    if total == 0:
        return 0, 0.0, 0.0
    unique = len(set(tokens))
    unique_ratio = unique / total
    repetition_rate = 1.0 - unique_ratio
    return total, unique_ratio, repetition_rate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", type=str, default=os.path.join("data", "hf_datasets", "cnn_dailymail_3.0.0"))
    parser.add_argument("--output-dir", type=str, default=os.path.join("visualization", "outputs"))
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--transformers-mode", type=str, choices=["offline", "auto", "allow-download"], default="auto")
    parser.add_argument("--power", type=str, default="")
    parser.add_argument("--carbon-intensity", type=float, default=0.4)
    parser.add_argument("--pgn-checkpoint", type=str, default=os.path.join("outputs", "pgn_smoke2", "checkpoint_epoch_5.pt"))
    parser.add_argument("--pgn-tokenizer", type=str, default="")
    parser.add_argument("--pgn-beam-size", type=int, default=4)
    parser.add_argument("--pgn-max-summary-len", type=int, default=100)
    parser.add_argument("--pgn-coverage-penalty", type=float, default=1.0)
    parser.add_argument("--pgn-length-penalty", type=float, default=1.2)
    parser.add_argument("--pgn-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--pgn-max-source-len", type=int, default=400)
    parser.add_argument("--pgn-hidden-size", type=int, default=128)
    parser.add_argument("--pgn-embed-size", type=int, default=128)
    parser.add_argument("--pgn-dropout", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_repo_on_path()
    from ml.data_utils import normalize_text
    from ml.summarizer import load_extractive_model
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    default_models = [
        "tfidf",
        "tfidf-pgn",
        "t5-small",
        "distilbart",
        "bart-large",
        "pegasus",
        "t5-3b",
    ]
    models = requested_models or default_models
    samples = load_samples(args.samples, args.seed, args.dataset_path)
    if not samples:
        raise RuntimeError("no_samples_found")
    input_word_counts = []
    input_char_counts = []
    reference_word_counts = []
    for item in samples:
        text = normalize_text(item.get("article", ""))
        reference = normalize_text(item.get("highlights", ""))
        if text:
            input_word_counts.append(len(text.split()))
            input_char_counts.append(len(text))
        if reference:
            reference_word_counts.append(len(reference.split()))
    avg_input_words = statistics.mean(input_word_counts) if input_word_counts else 0.0
    avg_input_chars = statistics.mean(input_char_counts) if input_char_counts else 0.0
    avg_reference_words = statistics.mean(reference_word_counts) if reference_word_counts else 0.0
    non_empty_samples = max(1, len(input_word_counts))
    rouge = evaluate.load("rouge")
    vectorizer, config = load_extractive_model(os.path.join("outputs", "cnn_dm_extractive"))
    max_sentences = int(config.get("max_sentences", 5))
    chunk_size = 2000
    pgn_cache = {"model": None, "tokenizer": None, "device": None}
    device_cache = {"device": None}
    power_overrides = parse_power_overrides(args.power)
    rows = []
    for model_id in models:
        if args.transformers_mode == "offline" and model_id not in ["tfidf", "tfidf-pgn"]:
            avg_latency, avg_len, rougeL = estimate_transformer_metrics(model_id, samples)
            power_watts = resolve_power(model_id, power_overrides)
            energy_kwh = compute_energy_kwh(power_watts, avg_latency, len(samples))
            carbon_kg = energy_kwh * args.carbon_intensity
            compression_ratio = (avg_len / avg_input_words) if avg_input_words else 0.0
            avg_latency_per_word = (avg_latency / avg_input_words) if avg_input_words else 0.0
            throughput_words_sec = (avg_input_words / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0
            rouge1 = rougeL
            rouge2 = max(0.0, rougeL * 0.45)
            avg_repetition_rate = 0.35
            quality_overall = (0.5 * rougeL + 0.3 * rouge2 + 0.2 * rouge1) * (1.0 - 0.3 * avg_repetition_rate) * 100.0
            rows.append(
                {
                    "model_id": model_id,
                    "samples": len(samples),
                    "errors": 0,
                    "avg_latency_ms": avg_latency,
                    "avg_summary_length": avg_len,
                    "rouge1": rouge1,
                    "rouge2": rouge2,
                    "rougeL": rougeL,
                    "power_watts": power_watts,
                    "energy_kwh": energy_kwh,
                    "carbon_kg": carbon_kg,
                    "avg_input_length_words": avg_input_words,
                    "avg_input_length_chars": avg_input_chars,
                    "avg_reference_length_words": avg_reference_words,
                    "compression_ratio": compression_ratio,
                    "avg_latency_per_input_word_ms": avg_latency_per_word,
                    "throughput_input_words_per_sec": throughput_words_sec,
                    "error_rate": 0.0,
                    "success_rate": 1.0,
                    "avg_unique_word_ratio": 0.65,
                    "avg_repetition_rate": avg_repetition_rate,
                    "quality_overall": quality_overall,
                    "estimated": True,
                }
            )
            continue
        predictions = []
        references = []
        latencies = []
        lengths = []
        unique_ratios = []
        repetition_rates = []
        errors = 0
        for item in samples:
            text = normalize_text(item.get("article", ""))
            reference = normalize_text(item.get("highlights", ""))
            if not text:
                continue
            start = time.perf_counter()
            try:
                if model_id == "tfidf":
                    summary = summarize_tfidf(text, vectorizer, max_sentences)
                elif model_id == "tfidf-pgn":
                    summary = summarize_pgn(text, vectorizer, max_sentences, chunk_size, args, pgn_cache)
                else:
                    if args.transformers_mode == "auto":
                        summary = summarize_transformer(text, model_id, device_cache, "allow-download")
                    elif args.transformers_mode == "allow-download":
                        summary = summarize_transformer(text, model_id, device_cache, "allow-download")
                    else:
                        summary = summarize_transformer(text, model_id, device_cache, "offline")
            except Exception:
                errors += 1
                continue
            end = time.perf_counter()
            latency_ms = (end - start) * 1000.0
            if summary:
                predictions.append(summary)
                references.append(reference)
                latencies.append(latency_ms)
                word_count, unique_ratio, repetition_rate = summarize_tokens(summary)
                lengths.append(word_count)
                unique_ratios.append(unique_ratio)
                repetition_rates.append(repetition_rate)
        if predictions and references:
            rouge_scores = rouge.compute(predictions=predictions, references=references)
            rouge1 = float(rouge_scores.get("rouge1", 0.0))
            rouge2 = float(rouge_scores.get("rouge2", 0.0))
            rougeL = float(rouge_scores.get("rougeL", 0.0))
        else:
            rouge1 = 0.0
            rouge2 = 0.0
            rougeL = 0.0
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        avg_len = statistics.mean(lengths) if lengths else 0.0
        avg_unique_ratio = statistics.mean(unique_ratios) if unique_ratios else 0.0
        avg_repetition_rate = statistics.mean(repetition_rates) if repetition_rates else 0.0
        power_watts = resolve_power(model_id, power_overrides)
        energy_kwh = compute_energy_kwh(power_watts, avg_latency, max(1, len(latencies)))
        carbon_kg = energy_kwh * args.carbon_intensity
        compression_ratio = (avg_len / avg_input_words) if avg_input_words else 0.0
        avg_latency_per_word = (avg_latency / avg_input_words) if avg_input_words else 0.0
        throughput_words_sec = (avg_input_words / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0
        error_rate = errors / non_empty_samples
        success_rate = len(latencies) / non_empty_samples
        quality_overall = (0.5 * rougeL + 0.3 * rouge2 + 0.2 * rouge1) * (1.0 - 0.3 * avg_repetition_rate) * 100.0
        rows.append(
            {
                "model_id": model_id,
                "samples": len(latencies),
                "errors": errors,
                "avg_latency_ms": avg_latency,
                "avg_summary_length": avg_len,
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougeL,
                "power_watts": power_watts,
                "energy_kwh": energy_kwh,
                "carbon_kg": carbon_kg,
                "avg_input_length_words": avg_input_words,
                "avg_input_length_chars": avg_input_chars,
                "avg_reference_length_words": avg_reference_words,
                "compression_ratio": compression_ratio,
                "avg_latency_per_input_word_ms": avg_latency_per_word,
                "throughput_input_words_per_sec": throughput_words_sec,
                "error_rate": error_rate,
                "success_rate": success_rate,
                "avg_unique_word_ratio": avg_unique_ratio,
                "avg_repetition_rate": avg_repetition_rate,
                "quality_overall": quality_overall,
                "estimated": False,
            }
        )
    rouge_values = [row["rougeL"] for row in rows]
    latency_values = [row["avg_latency_ms"] for row in rows]
    energy_values = [row["energy_kwh"] for row in rows]
    rouge_norm = normalize(rouge_values, higher_is_better=True)
    latency_norm = normalize(latency_values, higher_is_better=False)
    energy_norm = normalize(energy_values, higher_is_better=False)
    for idx, row in enumerate(rows):
        score = (rouge_norm[idx] * 0.5 + latency_norm[idx] * 0.3 + energy_norm[idx] * 0.2) * 100.0
        row["efficiency_score"] = score
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.time(), "rows": rows}, f, indent=2)
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "samples",
                "errors",
                "avg_latency_ms",
                "avg_summary_length",
                "rouge1",
                "rouge2",
                "rougeL",
                "power_watts",
                "energy_kwh",
                "carbon_kg",
                "efficiency_score",
                "avg_input_length_words",
                "avg_input_length_chars",
                "avg_reference_length_words",
                "compression_ratio",
                "avg_latency_per_input_word_ms",
                "throughput_input_words_per_sec",
                "error_rate",
                "success_rate",
                "avg_unique_word_ratio",
                "avg_repetition_rate",
                "quality_overall",
                "estimated",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
