# mech-interp

Mechanistic interpretability of a small character-level GPT. The model is a
10.8M-parameter transformer trained on tiny-shakespeare (from Karpathy's
"Let's build GPT, from scratch"); this repo reverse-engineers what its
attention heads actually do.

## The model

| Property       | Value                    |
| -------------- | ------------------------ |
| Parameters     | 10.8M                    |
| Layers / heads | 6 / 6 (64 dims per head) |
| Embedding dim  | 384                      |
| Context length | 256                      |
| Vocabulary     | 65 characters            |
| Best val loss  | 1.4885 (step 4000)       |

`scripts/transformer.py` holds the model and its training loop. The training run
is guarded under `if __name__ == '__main__'`, so a notebook can
`from transformer import GPTLanguageModel, encode, decode` and get the classes
instantly without kicking off a ~40-minute retrain.

## Layout

```
scripts/transformer.py    # model + training loop
notebooks/interp_1.ipynb  # attention-pattern analysis
input.txt                 # tiny-shakespeare corpus
models/                   # weights + checkpoints — gitignored (see below)
Research Report.pdf       # write-up
```

## Weights aren't in the repo

`models/` is gitignored: `model.pt` (best-val weights), `meta.pt` (config +
vocab), and 28 `ckpt_*.pt` training snapshots total ~1.4 GB. Regenerate them
from scratch:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib
python scripts/transformer.py        # ~40 min on Apple-silicon (MPS)
```

This writes `meta.pt` once, `model.pt` whenever validation loss improves, and a
checkpoint per snapshot step (dense early, then every 250 steps) — the series
used to study how circuits form over training.

## Findings so far

**Previous-token heads form early and strong.** Heads L0H5 (~0.68) and L1H2
(~0.64) attend almost entirely to the immediately preceding token — the first
half of a copying/induction circuit.

**Measured induction is near zero — yet the model copies anyway.** On
repeated-random sequences the best induction score is tiny (L5H2 ≈ 0.017). But
the model clearly copies *novel* strings it never saw in training: prompted with
`ZYXQWOR:` it continues `…XQWOR:`. The copying behaviour is real even though the
standard induction probe barely registers it.

**Leading hypothesis: tokenization.** With a character vocabulary an induction
head's key is a single character — matching on "P" is meaningless, every "P" is
identical, so the match step has nothing distinctive to grab. BPE chunks (e.g.
`RUCH`) would give the head a key worth matching on. The planned test
(`induction_2`): rerun the *identical* model with BPE instead of char
tokenization — one variable changed — and see whether the induction curve lifts
off the floor.

`interp_1.ipynb` also traces head scores across the `ckpt_*` snapshots to see
*when* in training the previous-token heads appear.

## Method

The notebook registers forward hooks on each attention head to cache its
post-softmax attention pattern, scores heads for previous-token attention and
induction, and probes copying directly with held-out and fabricated names.
