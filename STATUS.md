# STATUS (hand-off, 27 Nov 2026). All work stopped.

## SUMMARY (where we are, 27 Nov)
- Public LB 0.73 (top public notebooks: 0.725-0.730; top 10 ~0.777; rank 1 0.813).
- Built: pool loader, metrics, head_cv.py, blend_eval.py, LoRA trainer, Kaggle runner,
  OpenRouter tool (no free model matches our labels).
- Results:
  - Run B holdout 0.046 was invalid (predict-order bug in transformers 5, now fixed and checked).
    Run B is diagnostics only and never goes into a submission.
  - Run P (Qwen3-8B QLoRA, pool only, proxy rows dropped): BA 0.452 (95% 0.434-0.470) on all
    2,699 official rows. 2.4 GPU h used; 22.1 h left this week (resets Sat 28 Nov 05:30 IST).
  - CPU yardstick: head + TF-IDF 0.482 -> + run P 0.567, gain +0.085 (95% +0.069 to +0.101).
    This gain may not carry over to Rohan's full stack. Not submitted.
  - Weakest labels after P: gpt-5.6-terra, gemini-3.7-flash, gpt-5.6-sol, kimi-k3. Top mix-ups:
    gemini 3.6/3.7/3.8-flash with each other, gpt-5.6 terra/sol/luna with each other.
  - Head CV of the PB115 head alone was 0.244.
  - gem37 switch gained +0.000 (default OFF).
  - The 22,941 proxy-flagged rows are DROPPED from every retrain (default ON).
  - None of the 18 labels is open-weight and small enough for 2x T4 (GLM-5.3 744B, GLM-5.3-Flash
    320B, Kimi K3 2.8T, Qwen3.8-Max 2.4T; the other 14 are closed). Sizes from news; HF unreachable.
- Lessons: top public notebooks are ensembles of shared public assets (Qwen3-14B/8B,
  Llama-3.1-8B, Qwen2.5-14B LoRAs, Qwen3-Embedding-8B head, Qwen3.8-27B head, DeBERTa,
  TF-IDF+SVC), weight soups, group decoding with an injectivity constraint + Sinkhorn,
  and match_weight pool sampling.

## STATE
- Running: nothing. Run P's adapter and OOF stay in the Kaggle notebook ayush-pool-p.
- PRs #1, #2, #4 stay open. Nothing merged. Sonar gate on #4 fails on a known false positive.
- "Read top notebooks" task cancelled.

## JANUARY PLAN (start Sat 2 Jan 2027; final deadline Wed 13 Jan)
- 2-3 Jan: list every public notebook at or above the top public score, and every public
  asset they load (owner/name, licence, size). Skip forks.
- 3-6 Jan: build one submission notebook that loads the best public assets and combines them
  (blend weights from CV where possible, group decoding, proxy rows excluded from anything we
  train). Time it on a 4,000-row dummy and keep it under 8 h (limit 9 h). Drop the slowest,
  lowest-gain parts first. Run P's adapter is a candidate asset for the submission notebook.
- 6-11 Jan: submit variants, changing one thing at a time. Treat LB gaps under 0.01 as noise.
- 12-13 Jan: pick two finals (best CV, best LB). No new ideas after 11 Jan.
- Rules: public assets only, licences checked, no Benchmarks proxy, no paid APIs,
  commits only as ayushanand27.

## HOW TO RESUME
- Tools branch: claude/project-thread-ts88fs-tools (PR #4). Actions workflow kaggle.yml:
  fetch, pool_train (needs Ayush's yes), head_cv, blend_p, scan.
- Tests: python -m pytest -q (134 pass).
