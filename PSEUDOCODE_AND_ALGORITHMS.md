# AI News Summariser - Pseudocode & Algorithm Details

## 1. Main Application Pseudocode

```
PSEUDOCODE: AI News Summarizer Main Application

INITIALIZATION:
————————————————————————————————————————
  GLOBAL vectorizer ← LOAD_MODEL("outputs/cnn_dm_extractive/model.pkl")
  GLOBAL pgn_model ← LOAD_MODEL("outputs/pgn_smoke/best_checkpoint.pt") [if enabled]
  GLOBAL db ← SQLITE_CONNECT("data/app.db")
  GLOBAL settings ← LOAD_CONFIG("settings.json")
  
  HTTP_HEADERS ← {
    "User-Agent": "Mozilla/5.0...",
    "Accept-Language": "en-US,en;q=0.9"
  }

MAIN LOOP - REQUEST HANDLING:
————————————————————————————————————————
  WHILE True:
    request ← RECEIVE_HTTP_REQUEST()
    
    TRY:
      route ← PARSE_ROUTE(request.path)
      
      CASE route OF:
        "/":                          RETURN render_template("index.html")
        "/api/summarize":             HANDLE_SUMMARIZE_REQUEST(request)
        "/api/history":               HANDLE_HISTORY_REQUEST(request)
        "/api/export":                HANDLE_EXPORT_REQUEST(request)
        "/api/settings":              HANDLE_SETTINGS_REQUEST(request)
        "/api/health":                RETURN {"status": "ok"}
        DEFAULT:                      RETURN 404 NOT FOUND
    
    EXCEPT Exception as e:
      LOG_ERROR(e)
      RETURN {"error": str(e), "status": 500}


FUNCTION: HANDLE_SUMMARIZE_REQUEST(request)
————————————————————————————————————————
  INPUT:
    request.json = {
      "input_type": "url" | "text" | "file" | "media",
      "input_data": <data>,
      "model": "auto" | "tfidf" | "pgn" | "bart",
      "max_sentences": 5,
      "include_analysis": True
    }
  
  OUTPUT:
    JSON: {
      "summary": <text>,
      "summary_type": "extractive" | "abstractive",
      "latency_ms": <time>,
      "analysis": {
        "sentiment": <score>,
        "entities": [<list>],
        "key_phrases": [<list>],
        "readability": <score>
      }
    }
  
  PROCEDURE:
    1. START_TIMER()
    
    2. // INPUT VALIDATION
       IF NOT validate_input(request.json) THEN
         RETURN HTTP_ERROR(400, "Invalid input")
    
    3. // INPUT ROUTING
       CASE request.json.input_type OF:
         "url":
           article ← FETCH_URL(request.json.input_data, HTTP_HEADERS)
           EXTRACT_ARTICLE_CONTENT(article)
         
         "text":
           article ← PARSE_TEXT(request.json.input_data)
         
         "file":
           IF FILE_TYPE == "PDF":
             text ← EXTRACT_PDF_TEXT(file_path)
           ELSE IF FILE_TYPE IN ["JPG", "PNG", "JPEG"]:
             text ← RUN_OCR(file_path)
           ELSE:
             text ← READ_FILE(file_path)
           article ← PARSE_TEXT(text)
         
         "media":
           IF FILE_TYPE IN ["MP3", "WAV", "M4A"]:
             audio ← LOAD_AUDIO(file_path)
             text ← TRANSCRIBE_AUDIO(audio)
           ELSE IF FILE_TYPE IN ["MP4", "MOV", "MKV"]:
             audio ← EXTRACT_AUDIO_FROM_VIDEO(file_path)
             text ← TRANSCRIBE_AUDIO(audio)
           article ← PARSE_TEXT(text)
    
    4. // PREPROCESSING
       normalized_text ← NORMALIZE_TEXT(article.text)
       
       IF SIMHASH_CHECK(normalized_text):  // Deduplication
         RETURN HTTP_ERROR(409, "Article already summarized")
       
       IF token_count(normalized_text) > 8000:
         chunks ← CHUNK_TEXT_BY_TOKENS(
           normalized_text,
           max_tokens=512,
           stride=256
         )
       ELSE:
         chunks ← [normalized_text]
    
    5. // MODEL SELECTION
       CASE request.json.model OF:
         "auto":
           IF token_count < 200:
             model ← "tfidf"
           ELSE:
             model ← "pgn"
         
         OTHERWISE:
           model ← request.json.model
    
    6. // STAGE 1: EXTRACTIVE SUMMARIZATION
       summaries ← []
       FOR chunk IN chunks:
         extracted ← SUMMARIZE_EXTRACTIVE_TFIDF(
           chunk,
           vectorizer,
           max_sentences=request.json.max_sentences
         )
         summaries.APPEND(extracted)
    
    IF model == "tfidf":
       final_summary ← CONCATENATE(summaries)
       summary_type ← "extractive"
       GOTO STEP_7
    
    7. // STAGE 2: ABSTRACTIVE SUMMARIZATION (Optional)
       IF model == "pgn" AND pgn_model IS NOT NULL:
         
         // Tokenize extracted summary
         token_ids ← pgn_model.tokenizer.encode(extracted_summary)
         token_ids ← TRUNCATE_OR_PAD(token_ids, max_len=512)
         
         // Beam search decoding
         decoded_ids ← BEAM_SEARCH_DECODE(
           pgn_model,
           token_ids,
           beam_width=3,
           max_length=128
         )
         
         final_summary ← pgn_model.tokenizer.decode(decoded_ids)
         summary_type ← "abstractive"
       
       ELSE IF model == "bart":
         // Load BART model (e.g., facebook/bart-large-cnn)
         bart_model ← LOAD_BART_CHECKPOINT()
         batch = {
           "input_ids": tokenize(extracted_summary),
           "attention_mask": create_attention_mask(input_ids)
         }
         outputs ← bart_model.generate(batch, max_length=128)
         final_summary ← tokenizer.decode(outputs[0])
         summary_type ← "abstractive"
       
       ELSE:
         LOG_WARN("PGN/BART model not available, using TF-IDF")
         final_summary ← extracted_summary
         summary_type ← "extractive"
    
    8. // ENRICHMENT & ANALYSIS
       IF request.json.include_analysis:
         
         sentiment_score ← TEXTBLOB_SENTIMENT(final_summary).polarity
         sentiment_label ← "positive" IF sentiment_score > 0.1
                          "negative" IF sentiment_score < -0.1
                          "neutral"  OTHERWISE
         
         entities ← NER_EXTRACT(final_summary)  // Using spaCy or other
         
         key_phrases ← EXTRACT_KEY_PHRASES(
           final_summary,
           vectorizer,
           top_k=5
         )
         
         readability ← FLESCH_KINCAID_SCORE(final_summary)
         
         analysis ← {
           "sentiment_score": sentiment_score,
           "sentiment_label": sentiment_label,
           "entities": entities,
           "key_phrases": key_phrases,
           "readability": readability
         }
       ELSE:
         analysis ← NULL
    
    9. // EVALUATION (if reference available)
       IF reference_summary IS PROVIDED:
         rouge_scores ← COMPUTE_ROUGE(final_summary, reference_summary)
         // ROUGE-1, ROUGE-2, ROUGE-L
    
    10. // PERSISTENCE TO DATABASE
        article_id ← GENERATE_ID(article.url, article.publish_date, hash)
        
        TRY:
          INSERT INTO articles VALUES (
            id=article_id,
            url=article.url,
            title=article.title,
            cleaned_text=normalized_text,
            fetched_at=NOW(),
            hash=HASH(normalized_text)
          )
          
          INSERT INTO summaries VALUES (
            id=GENERATE_UUID(),
            article_id=article_id,
            model_name=model,
            summary_text=final_summary,
            created_at=NOW(),
            latency_ms=ELAPSED_TIME()
          )
          
          IF analysis IS NOT NULL:
            INSERT INTO analysis VALUES (...)
          
          COMMIT()
        
        EXCEPT DatabaseError:
          LOG_ERROR("Database save failed, using file cache")
          SAVE_TO_FILE_CACHE(article_id, final_summary, analysis)
    
    11. // RESPONSE FORMATTING
        response ← {
          "success": True,
          "summary": final_summary,
          "summary_type": summary_type,
          "latency_ms": ELAPSED_TIME(),
          "article_id": article_id,
          "tokens": {
            "source": token_count(normalized_text),
            "summary": token_count(final_summary),
            "compression_ratio": compression_ratio
          }
        }
        
        IF analysis IS NOT NULL:
          response["analysis"] ← analysis
        
        IF rouge_scores IS NOT NULL:
          response["evaluation"] ← rouge_scores
        
        RETURN HTTP_200_JSON(response)
  
  EXCEPT Exception as e:
    LOG_ERROR(e, traceback=TRUE)
    
    // FALLBACK: Try TF-IDF only
    TRY:
      fallback_summary ← SUMMARIZE_EXTRACTIVE_TFIDF(
        article.text, vectorizer, max_sentences=5
      )
      RETURN HTTP_200_JSON({
        "summary": fallback_summary,
        "warning": "Using fallback model due to error",
        "error_type": type(e).__name__
      })
    EXCEPT:
      RETURN HTTP_500_ERROR("Summarization failed completely")

END FUNCTION
```

---

## 2. TF-IDF Sentence Extraction Pseudocode

```
PSEUDOCODE: TF-IDF Extractive Summarization

FUNCTION: SUMMARIZE_EXTRACTIVE_TFIDF(
  article_text: String,
  vectorizer: TfidfVectorizer,
  max_sentences: Integer = 5
) → Dictionary

  INPUT:
    article_text: Raw article to summarize
    vectorizer: Pre-trained TF-IDF vectorizer
    max_sentences: Number of sentences to extract

  OUTPUT:
    {
      "summary": "Extracted summary text",
      "sentences": ["S1", "S2", "S3", ...],
      "scores": [s1, s2, s3, ...],
      "latency_ms": elapsed_time
    }

  PROCEDURE:
    1. start_time ← CURRENT_TIME()
    
    2. // NORMALIZE INPUT
       cleaned_text ← NORMALIZE_TEXT(article_text)
       
       IF LENGTH(cleaned_text) == 0:
         RETURN {"summary": "", "latency_ms": 0}
    
    3. // SENTENCE TOKENIZATION
       TRY:
         FROM nltk.tokenize IMPORT sent_tokenize
         sentences ← sent_tokenize(cleaned_text)
       EXCEPT:
         // Fallback: Split by periods
         raw_sentences ← split_by_delimiter(cleaned_text, ".")
         sentences ← [s.STRIP() FOR s IN raw_sentences IF s.STRIP()]
    
    4. // FILTER EMPTY SENTENCES
       sentences ← [s FOR s IN sentences IF LENGTH(s) > 0]
       
       IF LENGTH(sentences) == 0:
         RETURN {"summary": "", "latency_ms": 0}
    
    5. // CHECK IF TRUNCATION NEEDED
       IF LENGTH(sentences) <= max_sentences:
         summary_text ← JOIN(sentences, ". ")
         RETURN {
           "summary": summary_text,
           "sentences": sentences,
           "scores": [1.0] * LENGTH(sentences),
           "latency_ms": (CURRENT_TIME() - start_time) * 1000
         }
    
    6. // VECTORIZE SENTENCES (TF-IDF)
       tfidf_matrix ← vectorizer.TRANSFORM(sentences)
       // Shape: (num_sentences, num_features)
       
       IF tfidf_matrix.SHAPE[1] == 0:
         // No features extracted
         RETURN {"summary": "", "latency_ms": elapsed_time}
    
    7. // COMPUTE SENTENCE SCORES
       scores ← []
       
       FOR i IN RANGE(tfidf_matrix.SHAPE[0]):
         row ← tfidf_matrix[i, :]  // Sparse vector for sentence i
         
         // Sum all non-zero elements (weighted term values)
         score ← SUM(row.TOARRAY())
         
         scores.APPEND(score)
       
       // scores = [s1, s2, ..., sn] where si = Σ tfidf(term_j, sentence_i)
    
    8. // RANK SENTENCES BY SCORE
       scored_indices ← LIST(
         (index, score) FOR (index, score) IN ENUMERATE(scores)
       )
       
       // Sort descending by score
       scored_indices.SORT(key=lambda x: x[1], reverse=TRUE)
       
       // scored_indices = [(best_idx, best_score), (2nd_idx, 2nd_score), ...]
    
    9. // SELECT TOP-K SENTENCES
       top_k_indices ← []
       
       FOR idx IN RANGE(min(max_sentences, LENGTH(scored_indices))):
         sentence_index ← scored_indices[idx][0]
         top_k_indices.APPEND(sentence_index)
       
       // top_k_indices now contains indices of top-5 sentences by score
    
    10. // PRESERVE ORIGINAL NARRATIVE FLOW
        // Sort selected indices by their original position in article
        top_k_indices.SORT()  // Sort numerically (ascending)
        
        // Now top_k_indices = [i1, i2, i3, i4, i5] where i1 < i2 < i3 < ...
        // This preserves the order sentences appeared in the original text
    
    11. // EXTRACT SELECTED SENTENCES
        selected_sentences ← []
        selected_scores ← []
        
        FOR idx IN top_k_indices:
          selected_sentences.APPEND(sentences[idx])
          selected_scores.APPEND(scores[idx])
    
    12. // CONSTRUCT SUMMARY
        summary_text ← JOIN(selected_sentences, ". ")
        
        // Ensure proper punctuation
        IF NOT summary_text.ENDSWITH("."):
          summary_text ← summary_text + "."
    
    13. // COMPUTE LATENCY
        elapsed_ms ← (CURRENT_TIME() - start_time) * 1000
    
    14. RETURN {
          "summary": summary_text,
          "sentences": selected_sentences,
          "scores": selected_scores,
          "indices": top_k_indices,
          "latency_ms": elapsed_ms,
          "compression_ratio": LENGTH(summary_text) / LENGTH(article_text)
        }

END FUNCTION


ALGORITHM: TF-IDF Weight Calculation (Inside Vectorizer)
————————————————————————————————————————
  FOR each unique term t IN vocabulary:
    
    FOR each sentence d IN corpus:
      
      // Term Frequency: count of term t in sentence d
      tf(t, d) ← COUNT(t, d)
      
      // Inverse Document Frequency
      df(t) ← number of documents containing t
      idf(t) ← log((N + 1) / (df(t) + 1)) + 1
        // N = total number of sentences
        // +1 smoothing prevents log(0) and division by zero
      
      // TF-IDF Weight
      tfidf(t, d) ← tf(t, d) * idf(t)
    
    // Store (term, idf) pair in vectorizer for reuse

  Sentence Score Computation (Inference):
    FOR each sentence s:
      words ← TOKENIZE(s)
      
      score ← 0
      FOR word IN words:
        IF word IN vectorizer.vocabulary:
          word_vector ← vectorizer.TRANSFORM([word])
          tfidf_value ← word_vector[word]  // Retrieved from pre-computed weights
          score ← score + tfidf_value
      
      sentence_score[s] ← score

```

---

## 3. Pointer-Generator Network (PGN) Pseudocode

```
PSEUDOCODE: Pointer-Generator Network Training & Inference

INITIALIZATION - BUILD PGN MODEL:
————————————————————————————————————————

CLASS PointerGeneratorNetwork:
  
  CONSTRUCTOR(config):
    config = {
      "vocab_size": 8000,
      "embedding_dim": 128,
      "hidden_dim": 256,
      "max_source_len": 512,
      "max_target_len": 128,
      "coverage_weight": 0.15
    }
    
    // Embeddings
    THIS.embedding ← Embedding(
      num_embeddings=vocab_size,
      embedding_dim=embedding_dim
    )
    
    // Encoder: Bidirectional LSTM
    THIS.encoder ← BiLSTM(
      input_size=embedding_dim,
      hidden_size=hidden_dim,
      num_layers=1,
      bidirectional=True,
      batch_first=True
    )
    
    // Decoder: LSTM
    THIS.decoder ← LSTMCell(
      input_size=embedding_dim,
      hidden_size=hidden_dim
    )
    
    // Attention: Bahdanau
    THIS.attention ← Attention(
      query_dim=hidden_dim,
      memory_dim=2*hidden_dim  // BiLSTM outputs 2*hidden
    )
    
    // Output projections
    THIS.pointer_gen_gate ← Linear(
      in_features=hidden_dim + 2*hidden_dim,  // [h_t; context]
      out_features=1
    )
    
    THIS.output_projection ← Linear(
      in_features=hidden_dim + 2*hidden_dim,
      out_features=vocab_size
    )
    
    // Coverage tracking
    THIS.coverage_weights ← None


FUNCTION: TRAIN_PGN(
  dataset: HFDataset,
  config: Dict,
  num_epochs: Integer,
  batch_size: Integer
) → Checkpoint

  PROCEDURE:
    
    1. model ← PointerGeneratorNetwork(config)
    2. optimizer ← Adam(model.PARAMETERS(), lr=0.001)
    3. scheduler ← LinearLR(optimizer, total_steps=num_epochs*num_batches)
    
    4. FOR epoch IN RANGE(1, num_epochs + 1):
         
         total_loss ← 0
         num_batches ← 0
         
         5. FOR batch IN train_dataloader(batch_size):
              
              // Batch format:
              // batch = {
              //   "input_ids": [batch_size, seq_len],
              //   "target_ids": [batch_size, tgt_len],
              //   "target_ids_extended": [batch_size, tgt_len] (with OOV)
              // }
              
              6. source_input_ids ← batch["input_ids"]
                 target_input_ids ← batch["target_ids"][:-1]  // Shift right (teacher forcing)
                 target_output_ids_extended ← batch["target_ids_extended"][1:]  // Shift left
                 oov_list ← batch["oov_ids"]  // OOV token mappings
              
              7. // ENCODING
                 source_embeds ← model.embedding(source_input_ids)
                 
                 encoder_output, (encoder_hidden, encoder_cell) ← model.encoder(source_embeds)
                 // encoder_output: [batch, seq_len, 2*hidden_dim]
                 // encoder_hidden: [2, batch, hidden_dim]
              
              8. // DECODER INITIALIZATION
                 decoder_hidden ← encoder_hidden[-1]  // Last layer, forward + backward
                 decoder_cell ← torch.ZEROS_LIKE(decoder_hidden)
                 decoder_input ← BOS_TOKEN_ID
                 
                 coverage ← torch.ZEROS(
                   batch_size, source_sequence_length
                 ) // Cumulative attention
              
              9. batch_loss ← 0
                 coverage_loss ← 0
              
              10. FOR t IN RANGE(target_sequence_length):
                     // t-th step of decoding
                     
                     11. // Embed target token
                         decoder_input_embed ← model.embedding(decoder_input)
                     
                     12. // Decoder LSTM step
                         h_t, c_t ← model.decoder(
                           decoder_input_embed,
                           (decoder_hidden, decoder_cell)
                         )
                         // h_t: [batch, hidden_dim]
                         // c_t: [batch, hidden_dim]
                     
                     13. // Attention Computation
                         attention_weights ← model.attention.compute(
                           query=h_t,  // [batch, hidden_dim]
                           memory=encoder_output,  // [batch, seq_len, 2*hidden_dim]
                           coverage=coverage
                         )
                         // attention_weights: [batch, seq_len]
                         
                         context_vector ← WEIGHTED_SUM(
                           encoder_output,
                           attention_weights
                         )
                         // context_vector: [batch, 2*hidden_dim]
                     
                     14. // Update coverage
                         coverage ← coverage + attention_weights
                         
                         // Coverage loss: penalize re-attending
                         step_coverage_loss ← 
                           SUM(MIN(attention_weights, coverage))
                         coverage_loss ← coverage_loss + step_coverage_loss
                     
                     15. // Pointer-Generator Gate
                         combined ← CONCAT(h_t, context_vector)
                         // [batch, hidden_dim + 2*hidden_dim]
                         
                         gate_logit ← model.pointer_gen_gate(combined)
                         gate_prob ← SIGMOID(gate_logit)
                         // gate_prob: [batch, 1]
                         // Close to 1 → COPY from source
                         // Close to 0 → GENERATE from vocab
                     
                     16. // Generate vocabulary distribution
                         logits ← model.output_projection(combined)
                         p_vocab ← SOFTMAX(logits)
                         // p_vocab: [batch, vocab_size]
                     
                     17. // Copy distribution (from attention)
                         p_copy ← attention_weights
                         // [batch, seq_len]
                     
                     18. // Extended vocabulary distribution
                         p_final ← torch.ZEROS(
                           batch_size,
                           max_vocab_size_extended
                         )
                         
                         // Copy vocabulary part
                         p_final[:, :vocab_size] ← 
                           gate_prob * p_vocab +
                           (1 - gate_prob) * p_copy[:, :vocab_size]
                         
                         // OOV part (copy only)
                         IF max_vocab_size_extended > vocab_size:
                           p_final[:, vocab_size:] ← 
                             (1 - gate_prob) * p_copy[:, vocab_size:]
                     
                     19. // Compute loss for this step
                         target_token ← target_output_ids_extended[:, t]
                         
                         step_nll_loss ← -LOG(
                           p_final[range(batch), target_token]
                         )  // Negative log-likelihood
                         
                         step_nll_loss ← MEAN(step_nll_loss)  // Average over batch
                         
                         batch_loss ← batch_loss + step_nll_loss
                     
                     20. // Update decoder state
                         decoder_hidden ← h_t
                         decoder_cell ← c_t
                         decoder_input ← target_token  // Teacher forcing
    
              21. // Total loss with coverage penalty
                  total_step_loss ← batch_loss / target_sequence_length
                  total_coverage_loss ← coverage_loss / target_sequence_length
                  final_loss ← total_step_loss + 
                               config["coverage_weight"] * total_coverage_loss
              
              22. // Backward pass
                  optimizer.ZERO_GRAD()
                  final_loss.BACKWARD()
                  
                  // Gradient Clipping
                  torch.nn.utils.clip_grad_norm(
                    model.PARAMETERS(),
                    max_norm=1.0
                  )
                  
                  optimizer.STEP()
              
              23. total_loss ← total_loss + final_loss
                  num_batches ← num_batches + 1
         
         24. // VALIDATION PHASE (every 500 batches)
             IF num_batches % 500 == 0:
               
               val_rouge_scores ← EVAL_ON_VALIDATION_SET(model)
               
               IF val_rouge_scores["rouge2"] > best_rouge2_score:
                 best_rouge2_score ← val_rouge_scores["rouge2"]
                 SAVE_CHECKPOINT(model, epoch, val_rouge_scores)
                 LOG("New best model saved!")
         
         25. AVERAGE_LOSS ← total_loss / num_batches
             LOG("Epoch", epoch, "Loss:", AVERAGE_LOSS)
             scheduler.STEP()
    
    26. RETURN best_checkpoint


FUNCTION: BEAM_SEARCH_DECODE(
  model: PointerGeneratorNetwork,
  source_input_ids: Tensor,
  beam_width: Integer = 3,
  max_decode_length: Integer = 128
) → Tensor

  INPUT:
    source_input_ids: [seq_len]
    beam_width: Number of beams (typically 3-5)
    max_decode_length: Maximum target sequence length

  OUTPUT:
    decoded_ids: Best sequence found by beam search

  PROCEDURE:
    
    1. // ENCODING (same as training)
       source_embed ← model.embedding(source_input_ids)
       encoder_output, (encoder_hidden, encoder_cell) ← 
         model.encoder(source_embed)
    
    2. // INITIALIZATION
       beams ← [
         {
           "sequence": [BOS_TOKEN_ID],
           "log_prob": 0.0,
           "decoder_hidden": encoder_hidden[-1],
           "decoder_cell": encoder_cell[-1],
           "coverage": torch.ZEROS(len(source_input_ids))
         }
       ]  // Single initial beam
       
       finished_beams ← []
    
    3. FOR step IN RANGE(max_decode_length):
         
         4. candidates ← []
         
         5. FOR beam IN beams:
              
              6. current_token ← beam["sequence"][-1]
                 h_t ← beam["decoder_hidden"]
                 c_t ← beam["decoder_cell"]
                 coverage ← beam["coverage"]
              
              7. // Decoder step
                 input_embed ← model.embedding(current_token)
                 h_new, c_new ← model.decoder(
                   input_embed, (h_t, c_t)
                 )
              
              8. // Attention & pointer-generator (same as training)
                 attention_weights ← model.attention.compute(
                   query=h_new,
                   memory=encoder_output,
                   coverage=coverage
                 )
                 context_vector ← WEIGHTED_SUM(encoder_output, attention_weights)
                 combined ← CONCAT(h_new, context_vector)
                 
                 gate_prob ← SIGMOID(model.pointer_gen_gate(combined))
                 p_vocab ← SOFTMAX(model.output_projection(combined))
                 
                 p_final ← 
                   gate_prob * p_vocab + 
                   (1 - gate_prob) * attention_weights
              
              9. // Get top-k candidates from p_final
                 log_probs ← LOG(p_final + 1e-10)
                 
                 top_k_log_probs, top_k_indices ← 
                   TOPK(log_probs, k=beam_width)
              
              10. FOR i IN RANGE(beam_width):
                    new_log_prob ← 
                      beam["log_prob"] + top_k_log_probs[i]
                    
                    new_sequence ← 
                      beam["sequence"] + [top_k_indices[i]]
                    
                    new_coverage ← 
                      coverage + attention_weights
                    
                    candidates.APPEND({
                      "sequence": new_sequence,
                      "log_prob": new_log_prob,
                      "decoder_hidden": h_new,
                      "decoder_cell": c_new,
                      "coverage": new_coverage,
                      "is_finished": (top_k_indices[i] == EOS_TOKEN_ID)
                    })
         
         11. // Sort candidates by log probability
             candidates.SORT(key=lambda x: x["log_prob"], reverse=TRUE)
         
         12. // Separate finished and active beams
             new_beams ← []
             
             FOR candidate IN candidates:
               IF candidate["is_finished"]:
                 finished_beams.APPEND(candidate)
               ELSE:
                 new_beams.APPEND(candidate)
               
               IF LEN(new_beams) == beam_width:
                 BREAK
             
             beams ← new_beams
         
         13. // Check stopping condition
             IF LEN(beams) == 0:  // All beams have finished
               BREAK
    
    14. // Select best finished beam or best active beam
        all_beams ← finished_beams + beams
        best_beam ← all_beams[0]  // Highest log probability
    
    15. RETURN best_beam["sequence"]

END FUNCTION

```

---

## 4. Data Preprocessing Pipeline Pseudocode

```
PSEUDOCODE: Text Preprocessing Pipeline

FUNCTION: PREPROCESS_ARTICLE(raw_text: String) → String

  INPUT:
    raw_text: HTML or raw plain text from article

  OUTPUT:
    cleaned_text: Preprocessed, normalized text

  PROCEDURE:
    
    1. // HTML CLEANING (if HTML)
       IF input contains HTML tags:
         html_content ← PARSE_HTML(raw_text)
         text ← EXTRACT_TEXT_FROM_HTML(html_content)
         
         // Remove script and style elements
         FOR tag IN ["script", "style"]:
           REMOVE_ALL_TAGS(tag)
         
         text ← GET_TEXT()
       ELSE:
         text ← raw_text
    
    2. // UNICODE NORMALIZATION
       text ← replace(text, "\u00a0", " ")  // Non-breaking space → space
       text ← replace(text, "\u2019", "'")  // Right single quotation mark
       text ← replace(text, "\u2013", "-")  // En dash
       text ← replace(text, "\u2014", "-")  // Em dash
    
    3. // CONTROL CHARACTER REMOVAL
       text ← regex_replace(text, r"[\u0000-\u001f]+", " ")
       
    4. // WHITESPACE NORMALIZATION
       // Collapse multiple spaces
       text ← regex_replace(text, r"\s+", " ")
       text ← STRIP()  // Remove leading/trailing
    
    5. // PUNCTUATION NORMALIZATION
       // Multiple periods → single period
       text ← regex_replace(text, r"[.]{2,}", ".")
       
       // Multiple commas → single comma
       text ← regex_replace(text, r"[,]{2,}", ",")
       
       // Add spaces around punctuation
       text ← regex_replace(text, r"\s*([,:;.!?])\s*", r" \1 ")
       
       // Final whitespace cleanup
       text ← regex_replace(text, r"\s+", " ")
       text ← STRIP()
    
    6. // CHARACTER CASE HANDLING
       // Keep original case for named entities, title case awareness
       // (For PGN, case is important)
    
    7. IF LENGTH(text) == 0:
       RETURN ""
    
    8. RETURN text

END FUNCTION


FUNCTION: DUPLICATE_DETECTION(text: String) → Boolean

  PURPOSE: Detect if article is a duplicate of previously seen content

  PROCEDURE:
    
    1. // METHOD 1: SHA-1 Hash (exact match)
       text_hash ← SHA1(text)
       
       IF text_hash IN database.hashes:
         RETURN TRUE  // Exact duplicate
    
    2. // METHOD 2: SimHash (near-duplicate detection)
       fingerprint ← SIMHASH(text, bits=64)
       
       FOR stored_fp IN database.fingerprints:
         distance ← HAMMING_DISTANCE(fingerprint, stored_fp)
         
         IF distance <= 3:  // Threshold
           RETURN TRUE  // Near-duplicate
    
    3. RETURN FALSE  // Unique

END FUNCTION


FUNCTION: SIMHASH(text: String, bits: Integer = 64) → Integer

  PURPOSE: Generate locality-sensitive hash for near-duplicate detection

  PROCEDURE:
    
    1. tokens ← REGEX_FINDALL(r"\w+", text.LOWER())
    
    2. v ← [0] * bits  // Integer accumulator array
    
    3. FOR token IN tokens:
         
         md5_hash ← MD5(token).to_hexadecimal()
         hash_int ← convert_to_integer(md5_hash)
         
         FOR bit_index IN RANGE(bits):
           IF (hash_int >> bit_index) & 1 == 1:
             v[bit_index] ← v[bit_index] + 1
           ELSE:
             v[bit_index] ← v[bit_index] - 1
    
    4. fingerprint ← 0
    
    5. FOR bit_index IN RANGE(bits):
         IF v[bit_index] > 0:
           fingerprint ← fingerprint | (1 << bit_index)
    
    6. RETURN fingerprint

END FUNCTION


FUNCTION: CHUNK_TEXT_BY_TOKENS(
  text: String,
  tokenizer: Tokenizer,
  max_tokens: Integer = 512,
  stride: Integer = 256
) → List[String]

  PURPOSE: Split long text into overlapping chunks (token-aware)

  PROCEDURE:
    
    1. cleaned_text ← NORMALIZE_TEXT(text)
    
    2. IF LENGTH(cleaned_text) == 0:
       RETURN []
    
    3. token_ids ← tokenizer.ENCODE(cleaned_text)
    
    4. IF LENGTH(token_ids) <= max_tokens:
       RETURN [cleaned_text]  // No chunking needed
    
    5. chunks ← []
    6. start ← 0
    
    7. WHILE start < LENGTH(token_ids):
         
         8. end ← MIN(start + max_tokens, LENGTH(token_ids))
         
         9. chunk_ids ← token_ids[start : end]
         
         10. chunk_text ← tokenizer.DECODE(chunk_ids)
         
         11. chunk_text ← NORMALIZE_TEXT(chunk_text)
         
         12. IF LENGTH(chunk_text) > 0:
             chunks.APPEND(chunk_text)
         
         13. // Check if at end
             IF end == LENGTH(token_ids):
               BREAK
             
         14. // Move start with overlap (stride)
             start ← MAX(end - stride, start + 1)
    
    15. RETURN chunks

END FUNCTION

```

---

## 5. Evaluation Metrics Computation Pseudocode

```
PSEUDOCODE: ROUGE Score Computation

FUNCTION: COMPUTE_ROUGE(
  hypothesis: String,
  reference: String
) → Dictionary

  PURPOSE: Compute ROUGE-1, ROUGE-2, ROUGE-L scores

  OUTPUT:
    {
      "rouge1": {"precision": p, "recall": r, "f_score": f},
      "rouge2": {"precision": p, "recall": r, "f_score": f},
      "rougeL": {"precision": p, "recall": r, "f_score": f}
    }

  PROCEDURE:
    
    1. hypothesis_tokens ← TOKENIZE(hypothesis)
    2. reference_tokens ← TOKENIZE(reference)
    
    3. // ROUGE-1: Unigram Overlap
       rouge1_precision ← COUNT_OVERLAPPING_UNIGRAMS(
         hypothesis_tokens, reference_tokens
       ) / LENGTH(hypothesis_tokens)
       
       rouge1_recall ← COUNT_OVERLAPPING_UNIGRAMS(
         hypothesis_tokens, reference_tokens
       ) / LENGTH(reference_tokens)
       
       rouge1_f ← 2 * (rouge1_precision * rouge1_recall) /
                   (rouge1_precision + rouge1_recall)
    
    4. // ROUGE-2: Bigram Overlap
       hypothesis_bigrams ← GENERATE_NGRAMS(hypothesis_tokens, n=2)
       reference_bigrams ← GENERATE_NGRAMS(reference_tokens, n=2)
       
       rouge2_precision ← COUNT_OVERLAPPING_NGRAMS(
         hypothesis_bigrams, reference_bigrams
       ) / LENGTH(hypothesis_bigrams)
       
       rouge2_recall ← COUNT_OVERLAPPING_NGRAMS(
         hypothesis_bigrams, reference_bigrams
       ) / LENGTH(reference_bigrams)
       
       rouge2_f ← 2 * (rouge2_precision * rouge2_recall) /
                   (rouge2_precision + rouge2_recall)
    
    5. // ROUGE-L: Longest Common Subsequence
       lcs_len ← LCS_LENGTH(hypothesis_tokens, reference_tokens)
       
       rougeL_precision ← lcs_len / LENGTH(hypothesis_tokens)
       rougeL_recall ← lcs_len / LENGTH(reference_tokens)
       
       rougeL_f ← 2 * (rougeL_precision * rougeL_recall) /
                   (rougeL_precision + rougeL_recall)
    
    6. RETURN {
         "rouge1": {
           "precision": rouge1_precision,
           "recall": rouge1_recall,
           "fScore": rouge1_f
         },
         "rouge2": {
           "precision": rouge2_precision,
           "recall": rouge2_recall,
           "fScore": rouge2_f
         },
         "rougeL": {
           "precision": rougeL_precision,
           "recall": rougeL_recall,
           "fScore": rougeL_f
         }
       }

END FUNCTION


FUNCTION: LCS_LENGTH(
  seq1: List,
  seq2: List
) → Integer

  PURPOSE: Compute length of longest common subsequence

  ALGORITHM: Dynamic Programming

  PROCEDURE:
    
    1. m ← LENGTH(seq1)
    2. n ← LENGTH(seq2)
    
    3. // Create DP table
       dp ← MATRIX(m + 1, n + 1, initialized=0)
    
    4. FOR i IN RANGE(1, m + 1):
         FOR j IN RANGE(1, n + 1):
           
           IF seq1[i-1] == seq2[j-1]:
             dp[i][j] ← dp[i-1][j-1] + 1
           ELSE:
             dp[i][j] ← MAX(dp[i-1][j], dp[i][j-1])
    
    5. RETURN dp[m][n]  // Length of LCS

END FUNCTION

```

---

## 6. End-to-End Workflow Pseudocode

```
PSEUDOCODE: Complete Article Processing Pipeline

FUNCTION: PROCESS_ARTICLE_END_TO_END(
  input_source: String,  // "url" | "text" | "file" | "media"
  input_data: String,
  config: Config
) → ProcessingResult

  PROCEDURE:
    
    START_TIME ← CURRENT_TIME()
    METRICS ← {}
    
    STEP_1_INPUT_ACQUISITION:
    ————————————————————————
    
    TRY:
      CASE input_source OF:
        "url":
          START_FETCH ← CURRENT_TIME()
          html ← FETCH_URL(
            input_data,
            timeout=10,
            headers=DEFAULT_HEADERS,
            retries=3
          )
          METRICS["fetch_time_ms"] ← (CURRENT_TIME() - START_FETCH) * 1000
          
          article ← PARSE_HTML_ARTICLE(html)  // Using newspaper3k
          article_text ← article.text
          article_title ← article.title
          article_authors ← article.authors
        
        "text":
          article_text ← input_data
          article_title ← EXTRACT_FIRST_LINE(input_data)
          article_authors ← []
        
        "file":
          FILE_EXT ← GET_FILE_EXTENSION(input_data)
          
          IF FILE_EXT IN [".pdf"]:
            article_text ← EXTRACT_PDF_TEXT(input_data)
          
          ELSE IF FILE_EXT IN [".jpg", ".png", ".jpeg"]:
            article_text ← RUN_OCR(input_data)
          
          ELSE:
            article_text ← READ_FILE_TEXT(input_data)
          
          article_title ← GET_FILENAME(input_data)
      
        "media":
          IF FILE_TYPE IN [".mp3", ".wav", ".m4a"]:
            audio ← LOAD_AUDIO(input_data)
          
          ELSE IF FILE_TYPE IN [".mp4", ".mov", ".mkv"]:
            audio ← EXTRACT_AUDIO_FROM_VIDEO(input_data)
          
          SAMPLE_RATE ← 16000  // Standard for speech
          article_text ← TRANSCRIBE_AUDIO(
            audio,
            sample_rate=SAMPLE_RATE,
            model="openai/whisper-base"
          )
          article_title ← "Transcribed Audio"
    
    EXCEPT FetchError as e:
      RETURN ERROR_RESULT("Failed to fetch article", e)
    EXCEPT FileNotFound as e:
      RETURN ERROR_RESULT("File not found", e)
    
    
    STEP_2_PREPROCESSING:
    ————————————————————————
    
    START_PREPROCESS ← CURRENT_TIME()
    
    // Normalize text
    cleaned_text ← NORMALIZE_TEXT(article_text)
    
    IF LENGTH(cleaned_text) == 0:
      RETURN ERROR_RESULT("No text extracted")
    
    // Deduplication check
    text_hash ← SHA1(cleaned_text)
    
    IF text_hash IN database.processed_hashes:
      RETURN ERROR_RESULT("Duplicate article", "HTTP_409")
    
    // Token chunking for long articles
    token_count ← COUNT_TOKENS(cleaned_text, tokenizer=TOKENIZER)
    
    IF token_count > 8000:
      chunks ← CHUNK_TEXT_BY_TOKENS(
        cleaned_text,
        tokenizer=TOKENIZER,
        max_tokens=512,
        stride=256
      )
    ELSE:
      chunks ← [cleaned_text]
    
    METRICS["preprocess_time_ms"] ← (CURRENT_TIME() - START_PREPROCESS) * 1000
    METRICS["token_count"] ← token_count
    METRICS["num_chunks"] ← LENGTH(chunks)
    
    
    STEP_3_EXTRACTIVE_SUMMARIZATION:
    ————————————————————————
    
    START_EXTRACT ← CURRENT_TIME()
    
    extracted_summaries ← []
    
    FOR chunk IN chunks:
      extracted ← SUMMARIZE_EXTRACTIVE_TFIDF(
        chunk,
        vectorizer=GLOBAL_VECTORIZER,
        max_sentences=config.max_summary_sentences
      )
      extracted_summaries.APPEND(extracted["summary"])
    
    extractive_summary ← CONCATENATE(extracted_summaries)
    
    METRICS["extract_time_ms"] ← (CURRENT_TIME() - START_EXTRACT) * 1000
    
    
    STEP_4_MODEL_SELECTION:
    ————————————————————————
    
    selected_model ← config.default_model
    
    IF selected_model == "auto":
      // Auto-select based on text length
      IF token_count < 200:
        selected_model ← "tfidf"
      ELSE:
        selected_model ← "pgn"
    
    
    STEP_5_ABSTRACTIVE_SUMMARIZATION:
    ————————————————————————
    
    final_summary ← extractive_summary
    summary_type ← "extractive"
    
    IF selected_model IN ["pgn", "bart", "pegasus"]:
      
      START_ABSTRACT ← CURRENT_TIME()
      
      TRY:
        IF selected_model == "pgn":
          // PGN Model
          model ← LOAD_PGN_MODEL()
          
          token_ids ← model.tokenizer.encode(extractive_summary)
          token_ids ← TRUNCATE_OR_PAD(token_ids, max_len=512)
          
          decoded_ids ← BEAM_SEARCH_DECODE(
            model,
            token_ids,
            beam_width=3,
            max_length=128
          )
          
          final_summary ← model.tokenizer.decode(decoded_ids)
        
        ELSE IF selected_model == "bart":
          // BART Model (from HuggingFace)
          model ← LOAD_BART_CHECKPOINT("facebook/bart-large-cnn")
          
          batch = {
            "input_ids": TOKENIZE(extractive_summary),
            "attention_mask": CREATE_ATTENTION_MASK(input_ids)
          }
          
          outputs ← model.GENERATE(batch, max_length=128)
          final_summary ← DECODE_TOKENS(outputs[0])
        
        summary_type ← "abstractive"
        METRICS["abstractive_time_ms"] ← (CURRENT_TIME() - START_ABSTRACT) * 1000
      
      EXCEPT ModelLoadError:
        LOG_ERROR("Model failed to load, using TF-IDF")
        METRICS["abstract_fallback"] ← TRUE
    
    
    STEP_6_ENRICHMENT_AND_ANALYSIS:
    ————————————————————————
    
    IF config.include_analysis:
      
      START_ANALYSIS ← CURRENT_TIME()
      
      analysis_results ← {
        "sentiment": {},
        "entities": [],
        "key_phrases": [],
        "readability": {}
      }
      
      // Sentiment Analysis
      blob ← TextBlob(final_summary)
      sentiment_polarity ← blob.sentiment.polarity  // -1 to 1
      
      IF sentiment_polarity > 0.1:
        sentiment_label ← "positive"
      ELSE IF sentiment_polarity < -0.1:
        sentiment_label ← "negative"
      ELSE:
        sentiment_label ← "neutral"
      
      analysis_results["sentiment"] ← {
        "score": sentiment_polarity,
        "label": sentiment_label
      }
      
      // Named Entity Recognition
      doc ← NLP(final_summary)  // Using spaCy
      entities ← [
        {"text": ent.text, "label": ent.label_}
        FOR ent IN doc.ents
      ]
      analysis_results["entities"] ← entities
      
      // Key Phrase Extraction (TF-IDF)
      sentences ← TOKENIZE_SENTENCES(final_summary)
      key_phrases ← EXTRACT_KEY_PHRASES(
        final_summary,
        vectorizer=GLOBAL_VECTORIZER,
        top_k=5
      )
      analysis_results["key_phrases"] ← key_phrases
      
      // Readability Scoring
      readability_score ← COMPUTE_FLESCH_KINCAID(final_summary)
      analysis_results["readability"] ← {
        "score": readability_score,
        "grade_level": "intermediate"
      }
      
      METRICS["analysis_time_ms"] ← (CURRENT_TIME() - START_ANALYSIS) * 1000
    
    ELSE:
      analysis_results ← NULL
    
    
    STEP_7_PERSISTENCE:
    ————————————————————————
    
    START_SAVE ← CURRENT_TIME()
    
    TRY:
      article_id ← GENERATE_ID(
        url=input_data IF input_source=="url" ELSE NULL,
        publish_date=article.publish_date IF available ELSE NOW(),
        hash=text_hash
      )
      
      // Insert article record
      INSERT INTO articles VALUES (
        id=article_id,
        url=input_data IF input_source=="url",
        title=article_title,
        cleaned_text=cleaned_text,
        fetched_at=NOW(),
        hash=text_hash,
        word_count=COUNT_WORDS(cleaned_text)
      )
      
      // Insert summary record
      INSERT INTO summaries VALUES (
        id=GENERATE_UUID(),
        article_id=article_id,
        model_name=selected_model,
        summary_text=final_summary,
        created_at=NOW(),
        latency_ms=TOTAL_ELAPSED_TIME()
      )
      
      // Insert analysis records
      IF analysis_results IS NOT NULL:
        INSERT INTO analysis VALUES (...)
      
      DATABASE.COMMIT()
      METRICS["persist_time_ms"] ← (CURRENT_TIME() - START_SAVE) * 1000
    
    EXCEPT DatabaseError as e:
      LOG_ERROR("Database save failed", e)
      // Fallback to file cache
      SAVE_TO_FILE_CACHE(article_id, final_summary, analysis_results)
      METRICS["persist_fallback"] ← TRUE
    
    
    STEP_8_RESPONSE_FORMATTING:
    ————————————————————————
    
    response ← {
      "success": TRUE,
      "article_id": article_id,
      "summary": final_summary,
      "summary_type": summary_type,
      "total_latency_ms": (CURRENT_TIME() - START_TIME) * 1000,
      "metrics": METRICS,
      "analysis": analysis_results
    }
    
    RETURN response

END FUNCTION

```

---

## Summary Table: Algorithm Complexity

| Algorithm | Time Complexity | Space Complexity | Notes |
|-----------|-----------------|------------------|-------|
| **TF-IDF Ranking** | O(n·m) | O(n·m) | n=sentences, m=features (~50k) |
| **BiLSTM Encoding** | O(seq·h²) | O(seq·h) | seq=512, h=256 |
| **Attention Computation** | O(seq²) | O(seq) | Quadratic in sequence length |
| **Beam Search Decode** | O(k·L·h²) | O(k·h) | k=beam width, L=length |
| **SimHash** | O(tokens·bits) | O(bits) | Fast near-duplicate detection |
| **ROUGE Computation** | O(n+m) | O(n+m) | Linear in token counts |

---

**Document Compiled:** 2026  
**For:** AI News Summariser Project  
**Purpose:** Serve as authoritative algorithm reference and implementation guide
