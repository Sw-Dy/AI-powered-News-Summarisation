import argparse
import json
import os
import re
import time
from dataclasses import dataclass

import evaluate
import sentencepiece as spm
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset

from ml.data_utils import normalize_text, set_seed, ensure_dir
from ml.summarizer import load_extractive_model, summarize_extractive


def clean_text_for_spm(text):
    text = normalize_text(text)
    if not text:
        return ""
    text = re.sub(r"[\u0000-\u001f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[,]{2,}", ",", text)
    text = re.sub(r"[:]{2,}", ":", text)
    text = re.sub(r"[;]{2,}", ";", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"\s*([,:;.!?])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_dataset_fields(name):
    if name == "cnn_dailymail":
        return "article", "highlights"
    return "document", "summary"


def load_hf_dataset(dataset_name, dataset_version, dataset_path):
    if dataset_path:
        return load_from_disk(dataset_path)
    return load_dataset(dataset_name, dataset_version)


def build_tfidf_dataset(dataset, dataset_name, vectorizer, max_sentences, max_train_samples, splits=None):
    source_key, target_key = get_dataset_fields(dataset_name)
    if splits is None:
        splits = list(dataset.keys())
    remove_columns = list(dataset["train"].column_names)

    def map_fn(example):
        article = clean_text_for_spm(example.get(source_key, ""))
        summary = clean_text_for_spm(example.get(target_key, ""))
        tfidf_summary = summarize_extractive(article, vectorizer, max_sentences=max_sentences)["summary"]
        return {"tfidf_text": clean_text_for_spm(tfidf_summary), "summary": summary}

    mapped = {}
    for split in splits:
        ds = dataset[split]
        if split == "train" and max_train_samples > 0:
            ds = ds.select(range(min(max_train_samples, len(ds))))
        mapped[split] = ds.map(map_fn, remove_columns=remove_columns)
    return mapped


def write_corpus_file(dataset, output_path, max_samples):
    split = dataset["train"]
    if max_samples > 0:
        split = split.select(range(min(max_samples, len(split))))
    with open(output_path, "w", encoding="utf-8") as f:
        for item in split:
            tfidf_text = clean_text_for_spm(item.get("tfidf_text", ""))
            summary = clean_text_for_spm(item.get("summary", ""))
            if tfidf_text:
                f.write(tfidf_text + "\n")
            if summary:
                f.write(summary + "\n")


def train_sentencepiece(corpus_path, output_dir, vocab_size):
    ensure_dir(output_dir)
    model_prefix = os.path.join(output_dir, "spm")
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=0.9995,
    )
    return model_prefix + ".model"


class SpTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.vocab_size = self.sp.get_piece_size()

    def encode_ids(self, text, add_bos=False, add_eos=False, max_len=None):
        ids = self.sp.encode(text, out_type=int)
        if max_len is not None:
            limit = max_len
            if add_bos:
                limit -= 1
            if add_eos:
                limit -= 1
            ids = ids[: max(limit, 0)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_pieces(self, text, add_bos=False, add_eos=False, max_len=None):
        pieces = self.sp.encode_as_pieces(text)
        if max_len is not None:
            limit = max_len
            if add_bos:
                limit -= 1
            if add_eos:
                limit -= 1
            pieces = pieces[: max(limit, 0)]
        if add_bos:
            pieces = [self.sp.id_to_piece(self.bos_id)] + pieces
        if add_eos:
            pieces = pieces + [self.sp.id_to_piece(self.eos_id)]
        ids = [self.sp.piece_to_id(p) for p in pieces]
        return ids, pieces

    def decode_ids(self, ids):
        ids = [i for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.sp.decode(ids)


def build_extended_ids(tokenizer, ids, pieces):
    oovs = []
    ext_ids = []
    for token_id, piece in zip(ids, pieces):
        if token_id == tokenizer.unk_id:
            if piece not in oovs:
                oovs.append(piece)
            ext_ids.append(tokenizer.vocab_size + oovs.index(piece))
        else:
            ext_ids.append(token_id)
    return ext_ids, oovs


def map_target_ids(tokenizer, ids, pieces, oovs):
    ext_ids = []
    for token_id, piece in zip(ids, pieces):
        if token_id == tokenizer.unk_id:
            if piece in oovs:
                ext_ids.append(tokenizer.vocab_size + oovs.index(piece))
            else:
                ext_ids.append(tokenizer.unk_id)
        else:
            ext_ids.append(token_id)
    return ext_ids


class PgnDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_source_len, max_target_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source_text = normalize_text(item.get("tfidf_text", ""))
        target_text = normalize_text(item.get("summary", ""))
        source_ids, source_pieces = self.tokenizer.encode_pieces(
            source_text, add_bos=False, add_eos=True, max_len=self.max_source_len
        )
        target_ids, target_pieces = self.tokenizer.encode_pieces(
            target_text, add_bos=False, add_eos=True, max_len=self.max_target_len
        )
        source_ext_ids, oovs = build_extended_ids(self.tokenizer, source_ids, source_pieces)
        target_ext_ids = map_target_ids(self.tokenizer, target_ids, target_pieces, oovs)
        decoder_input = [self.tokenizer.bos_id] + target_ids[:-1]
        return {
            "source_ids": source_ids,
            "source_ext_ids": source_ext_ids,
            "target_ids": target_ids,
            "target_ext_ids": target_ext_ids,
            "decoder_input": decoder_input,
            "oovs": oovs,
            "tfidf_text": source_text,
            "summary": target_text,
        }


def collate_fn(batch, pad_id):
    max_source = max(len(x["source_ids"]) for x in batch)
    max_target = max(len(x["target_ids"]) for x in batch)
    max_oov = max(len(x["oovs"]) for x in batch)
    source_ids = []
    source_ext_ids = []
    target_ids = []
    target_ext_ids = []
    decoder_input = []
    source_lens = []
    oovs = []
    for item in batch:
        source_lens.append(len(item["source_ids"]))
        source_ids.append(item["source_ids"] + [pad_id] * (max_source - len(item["source_ids"])))
        source_ext_ids.append(item["source_ext_ids"] + [pad_id] * (max_source - len(item["source_ext_ids"])))
        target_ids.append(item["target_ids"] + [pad_id] * (max_target - len(item["target_ids"])))
        target_ext_ids.append(item["target_ext_ids"] + [pad_id] * (max_target - len(item["target_ext_ids"])))
        decoder_input.append(item["decoder_input"] + [pad_id] * (max_target - len(item["decoder_input"])))
        oovs.append(item["oovs"])
    return {
        "source_ids": torch.tensor(source_ids, dtype=torch.long),
        "source_ext_ids": torch.tensor(source_ext_ids, dtype=torch.long),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "target_ext_ids": torch.tensor(target_ext_ids, dtype=torch.long),
        "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
        "source_lens": torch.tensor(source_lens, dtype=torch.long),
        "max_oov": max_oov,
        "oovs": oovs,
    }


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_ids, lengths):
        embeddings = self.dropout(self.embedding(input_ids))
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        h = torch.cat([h[-2], h[-1]], dim=1)
        c = torch.cat([c[-2], c[-1]], dim=1)
        h = torch.tanh(self.reduce_h(h)).unsqueeze(0)
        c = torch.tanh(self.reduce_c(c)).unsqueeze(0)
        return outputs, (h, c)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, dec_dim)
        self.W_c = nn.Linear(1, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_state, coverage, mask):
        enc_features = self.W_h(enc_outputs)
        dec_features = self.W_s(dec_state).unsqueeze(1)
        cov_features = self.W_c(coverage.unsqueeze(-1))
        scores = self.v(torch.tanh(enc_features + dec_features + cov_features)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_dist = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_dist.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn_dist


class PGNDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size + enc_dim, hidden_size, num_layers=1, batch_first=True)
        self.attention = BahdanauAttention(enc_dim, hidden_size)
        self.p_gen_linear = nn.Linear(enc_dim + hidden_size + embed_size, 1)
        self.vocab_linear = nn.Linear(enc_dim + hidden_size, vocab_size)

    def step(self, input_ids, state, enc_outputs, enc_mask, coverage, prev_context, p_gen_bias):
        emb = self.dropout(self.embedding(input_ids))
        lstm_input = torch.cat([emb, prev_context], dim=1).unsqueeze(1)
        output, new_state = self.lstm(lstm_input, state)
        dec_state = output.squeeze(1)
        context, attn = self.attention(enc_outputs, dec_state, coverage, enc_mask)
        p_gen_logits = self.p_gen_linear(torch.cat([context, dec_state, emb], dim=1))
        p_gen = torch.sigmoid(p_gen_logits + p_gen_bias)
        p_gen = torch.clamp(p_gen, min=0.1, max=0.9)
        vocab_dist = torch.softmax(self.vocab_linear(torch.cat([context, dec_state], dim=1)), dim=1)
        return vocab_dist, attn, p_gen, new_state, context


class PointerGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, dropout)
        self.decoder = PGNDecoder(vocab_size, embed_size, hidden_size, hidden_size * 2, dropout)

    def forward(self, batch, device, lambda_cov, p_gen_bias):
        source_ids = batch["source_ids"].to(device)
        source_ext_ids = batch["source_ext_ids"].to(device)
        target_ext_ids = batch["target_ext_ids"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        lengths = batch["source_lens"].to(device)
        max_oov = batch["max_oov"]
        enc_outputs, enc_state = self.encoder(source_ids, lengths)
        enc_mask = (source_ids != 0).float()
        coverage = torch.zeros_like(enc_mask)
        context = torch.zeros(source_ids.size(0), enc_outputs.size(2), device=device)
        state = enc_state
        total_loss = 0.0
        valid_steps = 0
        for t in range(decoder_input.size(1)):
            input_ids = decoder_input[:, t]
            vocab_dist, attn, p_gen, state, context = self.decoder.step(
                input_ids, state, enc_outputs, enc_mask, coverage, context, p_gen_bias
            )
            final_dist = calc_final_dist(vocab_dist, attn, p_gen, source_ext_ids, max_oov, self.encoder.embedding.num_embeddings)
            target_ids = target_ext_ids[:, t]
            step_loss = -torch.log(final_dist.gather(1, target_ids.unsqueeze(1)).squeeze(1) + 1e-12)
            cov_loss = torch.min(attn, coverage).sum(1)
            coverage = coverage + attn
            mask = (target_ids != 0).float()
            step_loss = (step_loss + lambda_cov * cov_loss) * mask
            total_loss = total_loss + step_loss.sum()
            valid_steps = valid_steps + mask.sum()
        return total_loss / torch.clamp(valid_steps, min=1.0)


def calc_final_dist(vocab_dist, attn_dist, p_gen, source_ext_ids, max_oov, vocab_size):
    if max_oov > 0:
        extra_zeros = torch.zeros(vocab_dist.size(0), max_oov, device=vocab_dist.device)
        vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
    vocab_dist = p_gen * vocab_dist
    attn_dist = (1 - p_gen) * attn_dist
    final_dist = vocab_dist.clone()
    final_dist.scatter_add_(1, source_ext_ids, attn_dist)
    return final_dist


@dataclass
class TrainConfig:
    dataset_name: str
    dataset_version: str
    dataset_path: str
    tfidf_model_dir: str
    output_dir: str
    max_train_samples: int
    max_sentences: int
    vocab_size: int
    max_source_len: int
    max_target_len: int
    hidden_size: int
    embed_size: int
    dropout: float
    batch_size: int
    grad_accum_steps: int
    epochs: int
    lr: float
    seed: int
    spm_corpus_samples: int
    p_gen_bias: float
    p_gen_bias_epochs: int


def build_train_config(args):
    return TrainConfig(
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        tfidf_model_dir=args.tfidf_model_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_sentences=args.max_sentences,
        vocab_size=args.vocab_size,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        hidden_size=args.hidden_size,
        embed_size=args.embed_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        spm_corpus_samples=args.spm_corpus_samples,
        p_gen_bias=args.p_gen_bias,
        p_gen_bias_epochs=args.p_gen_bias_epochs,
    )


def save_checkpoint(output_dir, epoch, model, optimizer, tokenizer_path):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer_path": tokenizer_path,
            "epoch": epoch,
        },
        path,
    )
    return path


def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def train_model(args):
    config = build_train_config(args)
    set_seed(config.seed)
    ensure_dir(config.output_dir)
    dataset = load_hf_dataset(config.dataset_name, config.dataset_version, config.dataset_path)
    vectorizer, _ = load_extractive_model(config.tfidf_model_dir)
    tfidf_dataset = build_tfidf_dataset(
        dataset,
        config.dataset_name,
        vectorizer,
        config.max_sentences,
        config.max_train_samples,
        splits=["train", "validation"],
    )
    corpus_path = os.path.join(config.output_dir, "spm_corpus.txt")
    write_corpus_file(tfidf_dataset, corpus_path, config.spm_corpus_samples)
    tokenizer_path = train_sentencepiece(corpus_path, config.output_dir, config.vocab_size)
    tokenizer = SpTokenizer(tokenizer_path)
    train_set = PgnDataset(tfidf_dataset["train"], tokenizer, config.max_source_len, config.max_target_len)
    val_set = PgnDataset(tfidf_dataset["validation"], tokenizer, config.max_source_len, config.max_target_len)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointerGenerator(tokenizer.vocab_size, config.embed_size, config.hidden_size, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    lambda_cov = 1.0
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        p_gen_bias = config.p_gen_bias if epoch <= config.p_gen_bias_epochs else 0.0
        p_gen_bias = torch.tensor(p_gen_bias, device=device).view(1, 1)
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            loss = model(batch, device, lambda_cov, p_gen_bias)
            loss = loss / config.grad_accum_steps
            loss.backward()
            if step % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item()
        val_loss = evaluate_loss(model, val_loader, device, lambda_cov)
        save_checkpoint(config.output_dir, epoch, model, optimizer, tokenizer_path)
        print(json.dumps({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss}))
    train_time = time.time() - start_time
    stats = report_resources(config.output_dir, train_time)
    with open(os.path.join(config.output_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(stats))
    return tokenizer_path


def evaluate_loss(model, loader, device, lambda_cov):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            p_gen_bias = torch.tensor(0.0, device=device).view(1, 1)
            loss = model(batch, device, lambda_cov, p_gen_bias)
            total += loss.item()
            count += 1
    if count == 0:
        return 0.0
    return total / count


def beam_search(model, tokenizer, batch, device, beam_size, max_len, coverage_penalty, length_penalty, no_repeat_ngram_size):
    model.eval()
    source_ids = batch["source_ids"].to(device)
    source_ext_ids = batch["source_ext_ids"].to(device)
    lengths = batch["source_lens"].to(device)
    max_oov = batch["max_oov"]
    enc_outputs, enc_state = model.encoder(source_ids, lengths)
    enc_mask = (source_ids != 0).float()
    beams = [
        {
            "tokens": [tokenizer.bos_id],
            "log_prob": 0.0,
            "state": enc_state,
            "context": torch.zeros(source_ids.size(0), enc_outputs.size(2), device=device),
            "coverage": torch.zeros_like(enc_mask),
            "cov_penalty": 0.0,
            "ended": False,
        }
    ]
    def has_repeated_ngram(tokens, n):
        if n <= 0:
            return False
        seen = set()
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            if ng in seen:
                return True
            seen.add(ng)
        return False

    def length_norm(length, alpha):
        return ((5.0 + float(length)) / 6.0) ** alpha

    for _ in range(max_len):
        all_candidates = []
        for beam in beams:
            if beam["ended"]:
                all_candidates.append(beam)
                continue
            input_id = torch.tensor([beam["tokens"][-1]], device=device)
            p_gen_bias = torch.tensor(0.0, device=device).view(1, 1)
            vocab_dist, attn, p_gen, state, context = model.decoder.step(
                input_id, beam["state"], enc_outputs, enc_mask, beam["coverage"], beam["context"], p_gen_bias
            )
            final_dist = calc_final_dist(vocab_dist, attn, p_gen, source_ext_ids, max_oov, model.encoder.embedding.num_embeddings)
            topk_probs, topk_ids = torch.topk(final_dist, beam_size, dim=1)
            for prob, token_id in zip(topk_probs[0], topk_ids[0]):
                new_tokens = beam["tokens"] + [token_id.item()]
                if no_repeat_ngram_size > 0 and has_repeated_ngram(new_tokens, no_repeat_ngram_size):
                    continue
                cov_penalty = beam["cov_penalty"] + torch.min(attn, beam["coverage"]).sum().item()
                ended = token_id.item() == tokenizer.eos_id
                all_candidates.append(
                    {
                        "tokens": new_tokens,
                        "log_prob": beam["log_prob"] + torch.log(prob + 1e-12).item(),
                        "state": state,
                        "context": context,
                        "coverage": beam["coverage"] + attn,
                        "cov_penalty": cov_penalty,
                        "ended": ended,
                    }
                )
        if not all_candidates:
            for beam in beams:
                if beam["ended"]:
                    all_candidates.append(beam)
                    continue
                input_id = torch.tensor([beam["tokens"][-1]], device=device)
                p_gen_bias = torch.tensor(0.0, device=device).view(1, 1)
                vocab_dist, attn, p_gen, state, context = model.decoder.step(
                    input_id, beam["state"], enc_outputs, enc_mask, beam["coverage"], beam["context"], p_gen_bias
                )
                final_dist = calc_final_dist(
                    vocab_dist, attn, p_gen, source_ext_ids, max_oov, model.encoder.embedding.num_embeddings
                )
                topk_probs, topk_ids = torch.topk(final_dist, beam_size, dim=1)
                for prob, token_id in zip(topk_probs[0], topk_ids[0]):
                    new_tokens = beam["tokens"] + [token_id.item()]
                    cov_penalty = beam["cov_penalty"] + torch.min(attn, beam["coverage"]).sum().item()
                    ended = token_id.item() == tokenizer.eos_id
                    all_candidates.append(
                        {
                            "tokens": new_tokens,
                            "log_prob": beam["log_prob"] + torch.log(prob + 1e-12).item(),
                            "state": state,
                            "context": context,
                            "coverage": beam["coverage"] + attn,
                            "cov_penalty": cov_penalty,
                            "ended": ended,
                        }
                    )
        all_candidates.sort(
            key=lambda x: (x["log_prob"] - coverage_penalty * x["cov_penalty"]) / length_norm(len(x["tokens"]), length_penalty),
            reverse=True,
        )
        beams = all_candidates[:beam_size]
        if all(b["ended"] for b in beams):
            break
    if not beams:
        return [tokenizer.eos_id]
    return beams[0]["tokens"]


def decode_tokens(tokenizer, tokens, oovs):
    pieces = []
    for token_id in tokens:
        if token_id in {tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id}:
            continue
        if token_id < tokenizer.vocab_size:
            pieces.append(tokenizer.sp.id_to_piece(token_id))
        else:
            oov_index = token_id - tokenizer.vocab_size
            if 0 <= oov_index < len(oovs):
                pieces.append(oovs[oov_index])
            else:
                pieces.append(tokenizer.sp.id_to_piece(tokenizer.unk_id))
    return tokenizer.sp.decode_pieces(pieces).strip()


def generate_summary(model, tokenizer, item, device, beam_size, max_len, coverage_penalty, length_penalty, no_repeat_ngram_size):
    batch = collate_fn([item], tokenizer.pad_id)
    tokens = beam_search(
        model, tokenizer, batch, device, beam_size, max_len, coverage_penalty, length_penalty, no_repeat_ngram_size
    )
    return decode_tokens(tokenizer, tokens, batch["oovs"][0])


def evaluate_rouge(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    tokenizer_path = args.tokenizer_path or checkpoint.get("tokenizer_path", "")
    if not tokenizer_path:
        raise SystemExit("tokenizer_path is required for evaluation")
    tokenizer = SpTokenizer(tokenizer_path)
    model = PointerGenerator(tokenizer.vocab_size, args.embed_size, args.hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = load_hf_dataset(args.dataset_name, args.dataset_version, args.dataset_path)
    vectorizer, _ = load_extractive_model(args.tfidf_model_dir)
    tfidf_dataset = build_tfidf_dataset(
        dataset, args.dataset_name, vectorizer, args.max_sentences, args.max_eval_samples, splits=[args.eval_split]
    )
    eval_split = tfidf_dataset[args.eval_split]
    if args.max_eval_samples > 0:
        eval_split = eval_split.select(range(min(args.max_eval_samples, len(eval_split))))
    tokenizer_ds = PgnDataset(eval_split, tokenizer, args.max_source_len, args.max_target_len)
    rouge = evaluate.load("rouge")
    preds = []
    refs = []
    baseline_preds = []
    for item in tokenizer_ds:
        pred = generate_summary(
            model,
            tokenizer,
            item,
            device,
            args.beam_size,
            args.max_summary_len,
            args.coverage_penalty,
            args.length_penalty,
            args.no_repeat_ngram_size,
        )
        preds.append(pred)
        refs.append(clean_text_for_spm(item["summary"]) if "summary" in item else "")
        baseline_preds.append(clean_text_for_spm(item["tfidf_text"]) if "tfidf_text" in item else "")
    pgn_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    tfidf_scores = rouge.compute(predictions=baseline_preds, references=refs, use_stemmer=True)
    results = {
        "tfidf": tfidf_scores,
        "pgn": pgn_scores,
    }
    output_path = os.path.join(args.output_dir, "rouge_results.json")
    ensure_dir(args.output_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results))
    print(json.dumps(results))


def sanity_check(args):
    if not args.checkpoint_path:
        raise SystemExit("checkpoint_path is required for sanity mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    tokenizer_path = args.tokenizer_path or checkpoint.get("tokenizer_path", "")
    if not tokenizer_path:
        raise SystemExit("tokenizer_path is required for sanity mode")
    tokenizer = SpTokenizer(tokenizer_path)
    model = PointerGenerator(tokenizer.vocab_size, args.embed_size, args.hidden_size, args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    text = clean_text_for_spm(args.input_text)
    if not text:
        text = "The US and India discussed critical minerals cooperation."
    source_ids, source_pieces = tokenizer.encode_pieces(text, add_bos=False, add_eos=True, max_len=args.max_source_len)
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
    print(decode_tokens(tokenizer, tokens, oovs))


def report_resources(output_dir, train_time):
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    disk = dir_size(output_dir) / (1024 * 1024)
    return {"train_time_sec": train_time, "gpu_max_mem_mb": gpu_mem, "output_dir_mb": disk}


def dir_size(path):
    total = 0
    if not os.path.exists(path):
        return total
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            total += os.path.getsize(fp)
    return total


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="cnn_dailymail")
    parser.add_argument("--dataset_version", default="3.0.0")
    parser.add_argument("--dataset_path", default="data/hf_datasets/cnn_dailymail_3.0.0")
    parser.add_argument("--tfidf_model_dir", default="outputs/cnn_dm_extractive")
    parser.add_argument("--output_dir", default="outputs/pgn")
    parser.add_argument("--max_train_samples", type=int, default=120000)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    parser.add_argument("--max_sentences", type=int, default=5)
    parser.add_argument("--vocab_size", type=int, default=9000)
    parser.add_argument("--max_source_len", type=int, default=400)
    parser.add_argument("--max_target_len", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spm_corpus_samples", type=int, default=50000)
    parser.add_argument("--p_gen_bias", type=float, default=-1.5)
    parser.add_argument("--p_gen_bias_epochs", type=int, default=2)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_summary_len", type=int, default=100)
    parser.add_argument("--coverage_penalty", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--input_text", default="")
    parser.add_argument("--mode", default="train")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        evaluate_rouge(args)
    elif args.mode == "sanity":
        sanity_check(args)
    else:
        raise SystemExit("mode must be train, eval, or sanity")


if __name__ == "__main__":
    main()
