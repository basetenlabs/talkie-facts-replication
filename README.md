# talkie facts replication

Code and data for [How does talkie learn and forget facts?](https://labs.baseten.co) (Base Labs, 2026), which extends
[Can a Language Model Learn Facts Continually in Its Weights?](https://arxiv.org/abs/2607.11020).

We write twenty real 2011–2019 sports facts, one at a time, into
[talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it), a 13B model trained only on text published
before 1931. Each write trains a fresh rank-16 LoRA on one fact and merges it into the BF16 weights before the next.
After every write we score five held-out recall questions per fact. We then delete individual writes and relearn lost
facts to test whether forgotten facts are still in the weights.

This repository covers the main experiment (Figures 1, 2, 4 and 6). The scale-up experiments (Figures 3, 5C and 7) are
not included.

## Contents

| Path | What it is |
|---|---|
| `data/facts.json` | The twenty facts in write order: question, answer and aliases, three distractors, five held-out questions and a source |
| `data/train.jsonl` | The 480 training items, 24 per fact (six each of question, declarative, application and composition) |
| `data/recall_labels.jsonl` | Judged recall for all 100 held-out questions at all 21 checkpoints (base and after each write) |
| `data/relearning.json` | The twelve lost facts, their counterfactual answers and competing writes, and the relearning settings |
| `configs/judge_rubric.txt` | The frozen GPT-5-mini grading rubric |
| `talkie_facts_replication/` | Writing, evaluation, judging, deletion and relearning |
| `tests/` | CPU tests, including a check that every write uses the original run's training rows and sample order |

The twenty adapters from the main experiment, one per write, are released separately as safetensors (link). Together
with this code they rebuild every checkpoint exactly, so no merged models are released.

## Setup

Run from this repository's root. Results were produced with Python 3.12, Torch 2.7.1 (CUDA 12.8) and talkie at commit
`35317ba`, on one H200 with deterministic kernels. Other GPUs or versions may change results in the last bits.

```bash
python -m pip install -r requirements.txt
```

For the CPU tests only, install `requirements-cpu.txt` and run `python -m pytest -q`.

## Reproducing the results

Recall numbers from the released labels (CPU, seconds): 0% at base, 91% right after each fact's own write, 33% after
all twenty writes.

```bash
python -m talkie_facts_replication.summarize
```

Run the twenty writes and score recall after each one (one H200, a few hours). This writes `stage_XX.jsonl` files and
one adapter per write. To replay the released adapters instead of training, add `--adapters <dir>`.

```bash
python -m talkie_facts_replication.write --output runs/main
```

Grade the generated answers with GPT-5-mini (needs `OPENAI_API_KEY`), then summarize them:

```bash
python -m talkie_facts_replication.judge runs/main --output runs/main/labels.jsonl
python -m talkie_facts_replication.summarize runs/main/labels.jsonl
```

Delete a lost fact's own write, or the newer write that displaced it, after all twenty writes (Figures 4D and 6B). This
reports the mean summed log-probability of the true and competing answers over the five held-out questions:

```bash
python -m talkie_facts_replication.delete --adapters <dir> --output runs/delete.json
```

Relearn each lost fact from the final and the base model, with its true answer or a counterfactual one, and record the
number of updates needed to reach p = ½ (Figure 4B–C). The full grid is 12 facts × 4 arms × 3 seeds:

```bash
python -m talkie_facts_replication.relearn --adapters <dir> --output runs/relearn.json
```

Deletion and relearning hold one or two copies of the weights in CPU memory (about 26 GB each).

## Method notes

- **Training.** Each write is 96 AdamW updates at learning rate 5e-5, batch size 1, gradient clipping 1.0, on a fresh
  rank-16, α = 32 LoRA on all seven projections of all 40 blocks. The loss is the mean over completion tokens,
  including the native end token. Adapter initialization and sample order are seeded per write.
- **Merging.** Each write's FP32 update is added to the current BF16 weights, then rounded to BF16.
- **Deleting a write.** We subtract its exact effective change, the BF16 weights after its merge minus before, from
  the final model.
- **Grading.** Strict recall counts only answers the judge labels `CORRECT`.

The original runs used Baseten project `q04gj23`, job `w7xzjo3` (access required).
