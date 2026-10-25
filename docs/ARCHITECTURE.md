# Architecture Notes

## Control flow

```text
question
  -> decide_intent()
       -> GENERAL: provider chain (general)
       -> DOCUMENT / DOCUMENT_REASONING:
            -> HybridRAGPipeline.retrieve()
                 -> dense search
                 -> BM25 search
                 -> reciprocal rank fusion
                 -> independent evidence gates
            -> if insufficient: deterministic refusal, no LLM
            -> format evidence with [S#] IDs
            -> provider chain (document policy)
            -> validate citations
  -> ChatResult
```

## Separation of concerns

- `src/router/classifier.py`: pure query classification. It has no knowledge of
  providers, indexes, or embeddings.
- `src/orchestration/intent.py`: adds follow-up behavior to classification.
- `src/router/router.py`: applies privacy policy and provider fallback. It does
  not decide retrieval.
- `src/rag/pipeline.py`: retrieval and evidence gating. It does not generate.
- `src/orchestration/orchestrator.py`: the only layer that combines routing,
  retrieval, generation, history, and citations.
- `src/rag/index.py`: persistence and incremental build logic.
- `src/llm/base.py`: common provider contract.

## Retrieval invariants

1. Dense and BM25 always run for a document question.
2. RRF uses one-based ranks and stable `chunk_id`.
3. Raw dense and BM25 scores are never added.
4. A chunk in only one list contributes one rank term.
5. Duplicate hits inside one list use the best rank.
6. The fused ordering is deterministic.
7. Evidence is sent to the LLM only after an independent dense or BM25 gate
   passes.
8. If no evidence passes, the LLM is not called.

## Privacy invariants

1. `LOCAL_ONLY=true` removes every cloud provider from every chain.
2. `LOCAL_ONLY=true` rejects a non-loopback `OLLAMA_HOST`; remote Ollama is not
   considered local.
3. `ALLOW_CLOUD_RAG` is required separately before document evidence can leave
   the machine.
4. Cloud use is printed immediately before the provider call.
5. A local provider failure does not trigger a cloud fallback unless both cloud
   permissions are enabled.
6. General cloud prompts exclude previous document-grounded history.
7. Logs contain metadata and routing information, not prompt or chunk text.
8. API keys live in `.env`/environment and are excluded from `repr`.

## Index format

`storage/manifest.json` records:

- index version;
- embedding model and dimension;
- chunk size and overlap;
- document hashes and paths;
- per-document chunk IDs;
- build timestamp and counts.

The manifest is the compatibility boundary. Changing the embedding model or
chunk settings forces re-embedding. File hashes detect changed documents.

## Thresholds

`DENSE_MIN_SIMILARITY` is a cosine-similarity gate. `BM25_MIN_SCORE` is a raw
BM25 gate. `BM25_MIN_COVERAGE` is the fraction of content query tokens present in
the top BM25 chunk. The BM25 gates intentionally do not share a scale with dense
cosine. Defaults are calibrated as conservative starting points for a small
personal corpus, not as universal values.

## Failure behavior

- Missing cloud credentials: provider is not registered.
- Local Ollama unavailable: a clear error is returned; no cloud fallback in
  local-only mode.
- Unsupported/empty document: warning, other documents continue.
- No evidence: deterministic refusal; no generation call.
- Invalid citation IDs: removed and reported.
- Answer containing only invalid citations: treated as a provider error, not an
  empty successful answer.
- Embedding/index incompatibility: rebuild required.
