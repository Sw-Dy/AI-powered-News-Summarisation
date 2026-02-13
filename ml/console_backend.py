import argparse
import re

import requests
import torch
import validators
from newspaper import Article

from ml.pgn import SpTokenizer, PointerGenerator, build_extended_ids, beam_search, decode_tokens
from ml.summarizer import load_extractive_model, summarize_extractive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/cnn_dm_extractive")
    parser.add_argument("--url", default="")
    parser.add_argument("--use_pgn", action="store_true")
    parser.add_argument("--pgn_checkpoint", default="outputs/pgn_smoke/checkpoint_epoch_1.pt")
    parser.add_argument("--pgn_tokenizer", default="")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_summary_len", type=int, default=100)
    parser.add_argument("--coverage_penalty", type=float, default=1.0)
    parser.add_argument("--max_source_len", type=int, default=400)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def fetch_article(url):
    if not validators.url(url):
        raise ValueError("invalid_url")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def load_pgn(args, device):
    checkpoint = torch.load(args.pgn_checkpoint, map_location=device)
    tokenizer_path = args.pgn_tokenizer or checkpoint.get("tokenizer_path", "")
    if not tokenizer_path:
        raise ValueError("tokenizer_path_missing")
    tokenizer = SpTokenizer(tokenizer_path)
    model = PointerGenerator(tokenizer.vocab_size, args.embed_size, args.hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def summarize_with_pgn(tfidf_text, model, tokenizer, device, args):
    source_ids, source_pieces = tokenizer.encode_pieces(
        tfidf_text, add_bos=False, add_eos=True, max_len=args.max_source_len
    )
    source_ext_ids, oovs = build_extended_ids(tokenizer, source_ids, source_pieces)
    batch = {
        "source_ids": torch.tensor([source_ids], dtype=torch.long),
        "source_ext_ids": torch.tensor([source_ext_ids], dtype=torch.long),
        "source_lens": torch.tensor([len(source_ids)], dtype=torch.long),
        "max_oov": len(oovs),
        "oovs": [oovs],
    }
    tokens = beam_search(model, tokenizer, batch, device, args.beam_size, args.max_summary_len, args.coverage_penalty)
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


def main():
    args = parse_args()
    url = args.url.strip()
    if not url:
        url = input("Enter news article URL: ").strip()
    text = fetch_article(url)
    vectorizer, config = load_extractive_model(args.model_dir)
    max_sentences = int(config.get("max_sentences", 5))
    result = summarize_extractive(text=text, vectorizer=vectorizer, max_sentences=max_sentences)
    if args.use_pgn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_pgn(args, device)
        try:
            pgn_summary = summarize_with_pgn(result["summary"], model, tokenizer, device, args)
            if is_gibberish(pgn_summary):
                print(result["summary"])
            else:
                print(pgn_summary)
        except Exception:
            print(result["summary"])
    else:
        print(result["summary"])


if __name__ == "__main__":
    main()
