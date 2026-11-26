# STATUS

## 1. UPDATED
26 Nov 2026, 14:30 IST

## 2. DONE
- PR #1 workflow fix; PR #2 QLoRA trainer (4x loss and gradient bug fixed, probe passed), .pyc removed (2 files). Both draft, not merged.
- TF-IDF yardstick on the grouped holdout: A 0.293, B 0.389 [0.354, 0.422].
- head_cv.py (GitHub CPU): PB115 head alone 0.2438 [0.2293, 0.2590]; GLM pair recall 0.74-0.75, others 0.07-0.38. It is a GLM specialist inside the stack, not the 0.719 pipeline.
- 115 features = frozen GLM model stats (no labels), so CV is not leaky; feature choice may be tuned on train (not sure).
- blend_eval.py rebuilt: any number of OOF files, greedy 0.05 weights, report via compare(), cross-fit line.
- generate_data.py: OpenRouter :free only, /key budget, 15/min, served-model check; 119 tests pass in PR #4.
- Free models: 11 :free ids on OpenRouter today, 0 of our 18 labels match. Part B gives no rows.
- Gemini 3.7 pool rows: 3,012 dated 8 Oct or later, 0 before, 529 undated. TF-IDF B switch off -> on: 0.3889 -> 0.3889, +0.0000 [-0.0185, +0.0167] -> DROP. 3.7 recall falls 0.433 -> 0.033.

## 3. RUNNING
- ayushanand2327/ayush-pool-b: run B (Qwen3-8B QLoRA, official + pool cap 1500, 1 epoch, 2x T4). Started 26 Nov 12:40 IST, expected finish about 15:30 IST.
  Fetch: gh api -X POST repos/ayushanand27/name-that-llm/actions/workflows/kaggle.yml/dispatches -f ref=claude/project-thread-ts88fs-openpool -f 'inputs[action]=fetch' -f 'inputs[slug]=ayush-pool-b'
  (or Actions > kaggle-run > Run workflow: action fetch, slug ayush-pool-b). The log has holdout BA, per-label recall and top mix-ups.

## 4. WAITING ON AYUSH
- Gemini 3.7 switch fails the keep rule (and 3.7 recall collapses). Should I turn it off by default?
- The PB115 head is not the 0.719 stack, so its head CV can't be the keep-rule yardstick. Should I use a blend of OOF files we can build (PB115 head, TF-IDF, our LoRA) as the yardstick, and the LB alone for Rohan's full stack?
- (Final retrain already has your GO; it starts after B ends, if at least 3 GPU hours are left.)

## 5. NEXT
1. Phase 3 part A: list public datasets for this competition (CPU, report only).
2. Read thread 749455 for the host prompt style (Kaggle refused my fetch with "too many requests").
3. When B ends: report it, then push the final retrain (slug ayush-pool-final, run final, 2 GPUs, about 2.75 h).

## 6. GPU HOURS
About 3.8 h used, about 26.2 h left this week (estimate, B still running). Resets Sat 28 Nov.

## 7. DECISIONS LOG
- cut256 GLM test dropped: its own feature check failed (re-read features differed 39%).
- Run C dropped; "final" = B recipe with no holdout.
- Money is now $0: the paid kimi/qwen/claude generator is replaced by the OpenRouter :free version.
- CPU jobs run on GitHub (free) and print numbers only. Daily OpenRouter job not built: no free model matches a label.
- Background polling stopped (your no-poll rule); B is checked once when work allows.
