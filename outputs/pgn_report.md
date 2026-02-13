# TF-IDF Guided Pointer-Generator Network for CNN/DailyMail

## Architecture
The system is a two-stage hybrid summarizer. Stage 1 is a frozen TF-IDF extractive filter that selects top-k sentences from each article. Stage 2 is an abstractive Pointer-Generator Network trained from scratch on the reduced text.

The encoder is a single-layer BiLSTM with hidden size 128. The decoder is a single-layer LSTM with hidden size 128. Attention is Bahdanau additive attention with coverage. At each decoding step the model computes:

P(word) = p_gen * P_vocab(word) + (1 - p_gen) * P_copy(word)

where P_copy is the attention distribution over the source tokens projected into the extended vocabulary. A coverage vector accumulates attention over time and is used for both attention scoring and coverage loss.

## Training Methodology
Training data pairs are formed as:

{ "tfidf_text": "top-k sentences", "summary": "gold highlights" }

Tokenizer: SentencePiece BPE trained from scratch on TF-IDF-filtered inputs and gold summaries with vocab size 11k. No pretrained tokenizer is used. The model is trained with Adam (lr=0.001), batch size 4, gradient accumulation 4 (effective 16), dropout 0.3, and teacher forcing. Coverage loss is added with λ=1.0.

## Justification for No-Pretraining
The goal is to demonstrate low-resource neural summarization without pretrained language models or tokenizers. This aligns with constrained deployment environments and clarifies the contribution of explicit extractive filtering and pointer-based copying, rather than implicit knowledge from pretrained models.

## Limitations
- Training from scratch requires more epochs and data to reach strong ROUGE scores.
- Pointer copying operates at the subword level, which can limit faithful word-level copying when tokenization yields unknown pieces.
- With small training subsets, abstractive quality lags the TF-IDF baseline.

## Results
Smoke-test evaluation (50 validation samples, 1 epoch, 200 training samples):

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| TF-IDF | 0.1860 | 0.0436 | 0.1098 |
| TF-IDF + PGN | 0.0093 | 0.0000 | 0.0092 |

These results validate the pipeline end-to-end but do not reflect full-scale training. The configuration is set to train on 100k–150k examples with 5–7 epochs to reach the expected ROUGE-1 target of 34–36 and surpass the TF-IDF baseline.
