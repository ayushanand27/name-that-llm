# STATUS

## 1. UPDATED
27 Nov 2026, 05:50 IST

## 2. DONE
- Run P done (pool only, proxy dropped): see decisions log. test.csv on Kaggle has only 3 rows (hidden test), so test scoring needs a submission notebook.
- Run B (diagnostics only): holdout BA 0.0463 [0.0278, 0.0667] = bug, not the model. transformers 5 also length-groups the predict loader, so B's probabilities were shuffled against their rows. Reproduced on CPU (0.06 vs 0.97 after fix). B's per-label recall is unusable; run P's replaces it.
- Fix: in-order predict sampler + order check (direct forward on 16 rows must agree, or nothing is saved) + test-set scoring. 131 tests pass, PR #4.
- Switches per Ayush: drop_flag_kb_proxy default ON (all retrains), drop_gem37_after_oct8 default OFF.
- head_cv.py: PB115 head alone 0.2438 [0.2293, 0.2590] (GLM specialist, not the 0.719 stack). Features not leaky.
- blend_eval.py, generate_data.py (OpenRouter :free; 0 of 18 labels match), list_free_models.py, datasets_scan.py (70 found, 14 read).
- Gemini 3.7 switch: +0.0000 [-0.0185, +0.0167] -> DROP.

## 3. RUNNING
- Nothing.

## 4. WAITING ON AYUSH
- Yes/no: submission notebook = Rohan's stack + run P's adapter on the hidden test (P's 18 probs as head features). About 0.5 GPU h, I'm not sure (hidden test size unknown).

## 5. NEXT
1. If yes: build and push the submission notebook (uses P's adapter, no retrain).
2. Rescan the 15 rate-limited datasets (report only).

## 6. GPU HOURS
7.88 h used, 22.12 h left (P used 2.42 h). Resets Sat 28 Nov 05:30 IST.

## 7. DECISIONS LOG
- Run P: BA 0.4523 [0.4342, 0.4704] on all 2,699 rows. Yardstick (head+TF-IDF) 0.4820 -> +P 0.5668, gain +0.0848 [+0.0693, +0.1011] -> KEEP, passes +0.008. May not carry over to Rohan's full stack.
- Open-weight labels: none fit 2x T4 (GLM-5.3 744B, GLM-5.3-Flash 320B, Kimi K3 2.8T, Qwen3.8-Max 2.4T; rest closed).
- Yardstick = CPU blend (PB115 head, TF-IDF, our LoRA); LB judges Rohan's full stack only; submit only at blend gain >= +0.008.
- Proxy-flagged rows (22,941) dropped from every retrain; run B never goes into a submission.
- Old final retrain (with proxy rows) cancelled; final recipe now drops them.
- Money is $0: OpenRouter :free only; no daily job (no free model matches).
- No background polling; runs are fetched once when work allows.
