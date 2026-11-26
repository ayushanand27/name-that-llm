# STATUS

## 1. UPDATED
26 Nov 2026, 13:55 IST

## 2. DONE
- PR #1: workflow pushes notebooks to Kaggle (draft, not merged).
- PR #2: Open Pool QLoRA trainer; 4x loss and gradient bug fixed (Trainer kwargs), zero head, probe passed (loss 2.89 -> 2.59 at step 55).
- PR #2: committed .pyc files are removed (2 files) and .gitignore covers __pycache__/ and *.pyc.
- PR #4: generate_data.py (paid-API version) and blend_eval.py, 85 tests pass. Being reworked for the $0 mission.
- TF-IDF yardstick on the grouped holdout: A 0.293, B 0.389 [0.354, 0.422].

## 3. RUNNING
- ayushanand2327/ayush-pool-b: run B (Qwen3-8B QLoRA, official + pool cap 1500, 1 epoch, 2x T4). Started 26 Nov 12:40 IST, expected finish about 15:30 IST.
  Fetch: gh api -X POST repos/ayushanand27/name-that-llm/actions/workflows/kaggle.yml/dispatches -f ref=claude/project-thread-ts88fs-openpool -f 'inputs[action]=fetch' -f 'inputs[slug]=ayush-pool-b'
  (or Actions > kaggle-run > Run workflow: action fetch, slug ayush-pool-b). The log has holdout BA, per-label recall and top mix-ups.

## 4. WAITING ON AYUSH
- Nothing right now. (Final retrain already has your GO; it starts after B ends, if at least 3 GPU hours are left.)

## 5. NEXT
1. Step 0: find PB115's 115-feature train matrix, check whether it is out-of-fold, build head_cv.py with add_features() and compare().
2. Rebuild blend_eval.py on compare(), and generate_data.py for OpenRouter :free (dry run, /key budget, 15 req/min).
3. When B ends: report it, then push the final retrain (slug ayush-pool-final, run final, 2 GPUs, about 2.75 h).

## 6. GPU HOURS
About 3.8 h used, about 26.2 h left this week (estimate, B still running). Resets Sat 28 Nov.

## 7. DECISIONS LOG
- cut256 GLM test dropped: its own feature check failed (re-read features differed 39%).
- Run C dropped; "final" = B recipe with no holdout.
- Money is now $0: the paid kimi/qwen/claude generator is replaced by the OpenRouter :free version.
- Background polling stopped (your no-poll rule); B is checked once when work allows.
