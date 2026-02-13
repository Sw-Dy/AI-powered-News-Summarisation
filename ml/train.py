import argparse
import json
import os
import pickle
import time

from datasets import load_dataset, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.data_utils import normalize_text, set_seed


def build_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset_name", default=defaults.get("dataset_name", "cnn_dailymail"))
    parser.add_argument("--dataset_version", default=defaults.get("dataset_version", "3.0.0"))
    parser.add_argument("--dataset_path", default=defaults.get("dataset_path", ""))
    parser.add_argument("--output_dir", default=defaults.get("output_dir", "outputs/cnn_dm_extractive"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--max_train_samples", type=int, default=defaults.get("max_train_samples", 5000))
    parser.add_argument("--max_features", type=int, default=defaults.get("max_features", 50000))
    parser.add_argument("--min_df", type=int, default=defaults.get("min_df", 2))
    parser.add_argument("--max_sentences", type=int, default=defaults.get("max_sentences", 5))
    return parser


def parse_args():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", default="")
    base_args, _ = base.parse_known_args()
    defaults = {}
    if base_args.config:
        with open(base_args.config, "r", encoding="utf-8") as f:
            defaults = json.load(f)
    parser = build_parser(defaults)
    return parser.parse_args()


def get_dataset_fields(name):
    if name == "cnn_dailymail":
        return "article", "highlights"
    return "document", "summary"


def load_data(args):
    if args.dataset_path:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_name, args.dataset_version)
    return dataset


def collect_texts(dataset, source_key, max_samples):
    split = dataset["train"]
    if max_samples > 0:
        split = split.select(range(min(max_samples, len(split))))
    texts = []
    for item in split:
        text = normalize_text(item.get(source_key, ""))
        if text:
            texts.append(text)
    return texts


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_data(args)
    source_key, _ = get_dataset_fields(args.dataset_name)
    texts = collect_texts(dataset, source_key, args.max_train_samples)
    if not texts:
        raise SystemExit("No training texts found.")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features, min_df=args.min_df)
    vectorizer.fit(texts)
    model_path = os.path.join(args.output_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(vectorizer, f)
    metadata = {
        "model_type": "extractive_tfidf",
        "dataset_name": args.dataset_name,
        "dataset_version": args.dataset_version,
        "dataset_path": args.dataset_path,
        "max_train_samples": args.max_train_samples,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_sentences": args.max_sentences,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata))


if __name__ == "__main__":
    main()
