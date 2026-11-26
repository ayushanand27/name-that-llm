# STATUS

## 1. UPDATED
27 Nov 2026, 01:00 IST

## 2. DONE
- Run B (diagnostics only): holdout BA 0.0463 [0.0278, 0.0667] = bug, not the model. transformers 5 also length-groups the predict loader, so B's probabilities were shuffled against their rows. Reproduced on CPU (0.06 vs 0.97 after fix). B's per-label recall is unusable; run P's replaces it.
- Fix: in-order predict sampler + order check (direct forward on 16 rows must agree, or nothing is saved) + test-set scoring. 131 tests pass, PR #4.
- Switches per Ayush: drop_flag_kb_proxy default ON (all retrains), drop_gem37_after_oct8 default OFF.
- head_cv.py: PB115 head alone 0.2438 [0.2293, 0.2590] (GLM specialist, not the 0.719 stack). Features not leaky.
- blend_eval.py, generate_data.py (OpenRouter :free; 0 of 18 labels match), list_free_models.py, datasets_scan.py (70 found, 14 read).
- Gemini 3.7 switch: +0.0000 [-0.0185, +0.0167] -> DROP.

## 3. RUNNING
- ayushanand2327/ayush-pool-p: run P (Qwen3-8B QLoRA, pool only, proxy rows dropped, cap 1500, 1 epoch, 2x T4). Scores all 2,699 official rows (out-of-fold) and test. Started 27 Nov 00:51 IST, expected finish about 03:45 IST (~2.8 h).
  Fetch: gh api -X POST repos/ayushanand27/name-that-llm/actions/workflows/kaggle.yml/dispatches -f ref=claude/project-thread-ts88fs-tools -f 'inputs[action]=fetch' -f 'inputs[slug]=ayush-pool-p'

## 4. WAITING ON AYUSH
- Nothing right now. (Public datasets stay report-only; nothing gets added without your yes.)

## 5. NEXT
1. Fetch P: report BA on 2,699 rows, per-label recall, top 10 confusions.
2. Add P's 18 probabilities to the yardstick blend (PB115 head + TF-IDF + P) and test with the keep rule. Blend gain may not carry over to Rohan's full stack.
3. Rescan the 15 rate-limited datasets (report only: licence, pool overlap, proxy-made or not).

## 6. GPU HOURS
5.44 h used, 24.56 h left before P; about 21.7 h left after P. Resets Sat 28 Nov 05:30 IST.

## 7. DECISIONS LOG
- Yardstick = CPU blend (PB115 head, TF-IDF, our LoRA); LB judges Rohan's full stack only; submit only at blend gain >= +0.008.
- Proxy-flagged rows (22,941) dropped from every retrain; run B never goes into a submission.
- Old final retrain (with proxy rows) cancelled; final recipe now drops them.
- Money is $0: OpenRouter :free only; no daily job (no free model matches).
- No background polling; runs are fetched once when work allows.
