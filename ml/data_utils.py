import hashlib
import os
import random
import re

import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_text(text):
    if text is None:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def simhash(text, bits=64):
    tokens = re.findall(r"\w+", text.lower())
    v = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i, score in enumerate(v):
        if score > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a, b):
    return (a ^ b).bit_count()


def deduplicate_dataset(dataset, text_key, use_simhash=False, simhash_threshold=3):
    seen = set()

    def keep(example):
        text = normalize_text(example.get(text_key, ""))
        if not text:
            return False
        if use_simhash:
            fp = simhash(text)
            for existing in seen:
                if hamming_distance(existing, fp) <= simhash_threshold:
                    return False
            seen.add(fp)
            return True
        fp = hash_text(text)
        if fp in seen:
            return False
        seen.add(fp)
        return True

    return dataset.filter(keep)


def chunk_text_by_tokens(text, tokenizer, max_tokens, stride):
    text = normalize_text(text)
    if not text:
        return []
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(normalize_text(chunk_text))
        if end == len(token_ids):
            break
        start = max(end - stride, start + 1)
    return [c for c in chunks if c]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
