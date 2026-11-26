# STATUS

## 1. UPDATED
26 Nov 2026, 14:20 IST

## 2. DONE
- PR #1: workflow pushes notebooks to Kaggle (draft, not merged).
- PR #2: Open Pool QLoRA trainer; 4x loss and gradient bug fixed (Trainer kwargs), zero head, probe passed (loss 2.89 -> 2.59 at step 55).
- PR #2: committed .pyc files are removed (2 files) and .gitignore covers __pycache__/ and *.pyc.
- PR #4: generate_data.py (paid-API version) and blend_eval.py, 85 tests pass. Being reworked for the $0 mission.
- TF-IDF yardstick on the grouped holdout: A 0.293, B 0.389 [0.354, 0.422].
- head_cv.py (GitHub CPU): PB115 head alone 0.2438 [0.2293, 0.2590]; GLM pair recall 0.74-0.75, others 0.07-0.38. It is a GLM specialist inside the stack, not the 0.719 pipeline.
- 115 features = stats from two frozen GLM models (no labels used), so the CV is not leaky; feature choice may be tuned on train (not sure).
- blend_eval.py rebuilt: any number of OOF files, greedy 0.05 weights, report via compare(), cross-fit line.

## 3. RUNNING
- ayushanand2327/ayush-pool-b: run B (Qwen3-8B QLoRA, official + pool cap 1500, 1 epoch, 2x T4). Started 26 Nov 12:40 IST, expected finish about 15:30 IST.
  Fetch: gh api -X POST repos/ayushanand27/name-that-llm/actions/workflows/kaggle.yml/dispatches -f ref=claude/project-thread-ts88fs-openpool -f 'inputs[action]=fetch' -f 'inputs[slug]=ayush-pool-b'
  (or Actions > kaggle-run > Run workflow: action fetch, slug ayush-pool-b). The log has holdout BA, per-label recall and top mix-ups.

## 4. WAITING ON AYUSH
- The PB115 head is not the 0.719 stack, so its head CV can't be the keep-rule yardstick. Should I use a blend of OOF files we can build (PB115 head, TF-IDF, our LoRA) as the yardstick, and the LB alone for Rohan's full stack?
- (Final retrain already has your GO; it starts after B ends, if at least 3 GPU hours are left.)

## 5. NEXT
1. generate_data.py for OpenRouter :free (dry run, /key budget, 15 req/min) and list_free_models.py (CPU).
2. Pool fix: count gemini-3.7 rows dated 8 Oct or later, add drop_gem37_after_oct8 (CPU).
3. When B ends: report it, then push the final retrain (slug ayush-pool-final, run final, 2 GPUs, about 2.75 h).

## 6. GPU HOURS
About 3.8 h used, about 26.2 h left this week (estimate, B still running). Resets Sat 28 Nov.

## 7. DECISIONS LOG
- cut256 GLM test dropped: its own feature check failed (re-read features differed 39%).
- Run C dropped; "final" = B recipe with no holdout.
- Money is now $0: the paid kimi/qwen/claude generator is replaced by the OpenRouter :free version.
- head_cv action added (GitHub CPU, free). Artifacts can't be downloaded here, so CPU jobs print numbers only.
- Background polling stopped (your no-poll rule); B is checked once when work allows.
