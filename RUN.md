# RUN.md — scaled training to grow induction heads (`transformer_vast.py`)

Goal: train a GPT-2-small-scale model on real web text until **induction heads form**, then
find them with the existing `notebooks/interp_1.ipynb` / `notebooks/Interp_2.ipynb` tools.
The char and small-BPE runs both read induction ≈ chance; this run is the demonstration that
it lifts off the floor. **The science (the sweep, the heatmaps, finding the head) is done by
hand in the notebooks — not by this rig.**

- **Model:** GPT-2-small (n_layer 12, n_embd 768, n_head 12, block_size 512), BPE vocab 8192
- **Data:** ~1GB streamed OpenWebText subset → `data/vast/{train,val}.bin`
- **Hardware:** single RTX 4090 (24GB) on Vast.ai
- **Cost ceiling:** **$20 — stop the instance the moment the meter reaches it** (overnight 4090 ≈ $3–8, so this is headroom for provisioning/false-starts, not a target)

`models/vast/` and `data/vast/` are gitignored (mirrors `models/char`, `models/bpe`).

---

## 0. Decide first
- Dataset: OpenWebText (streamed). Change only if you also update `TEXT_COLUMN` in the prep script.
- A hard **cost ceiling** — write the number into the header above so the run is bounded by
  decision, not discovered mid-meter. Estimate: ~$0.30–0.50/hr for a 4090; overnight ≈ $3–8.

## 1. Prep data locally on the Mac (free, no GPU)
```bash
source .venv/bin/activate
pip install datasets numpy        # tokenizers + torch already installed

python scripts/prepare_vast_data.py --smoke   # ~2MB pipeline test, seconds — confirms bins load + get_batch is sane
python scripts/prepare_vast_data.py           # the real ~1GB subset (streams OWT, stops at the token budget)
```
This streams OpenWebText (never downloads the full ~40GB), trains the BPE tokenizer, writes
`data/vast/{train,val}.bin` + `models/vast/meta.pt`, and verifies the result. **Do the
`--smoke` run first** — if the pipeline is broken, find out in seconds, not after a 1GB tokenize.

## 2. Rent the 4090 on Vast
- Pick a 4090 / 24GB image with PyTorch + CUDA. Budget extra wall-clock for first-time provisioning.
- Get the repo + the prepared data + a venv onto it, e.g.:
```bash
rsync -avz --exclude models/char --exclude models/bpe \
  ./ user@<vast-host>:~/mech-interp/        # includes data/vast/*.bin and models/vast/meta.pt
# on the box:
pip install torch numpy datasets tokenizers
```
(You can instead re-run `prepare_vast_data.py` on the box, but copying the bins skips re-tokenizing.)

## 3. Smoke test on the GPU (~$0.50)
Edit `scripts/transformer_vast.py`: set `max_iters = 50`, then:
```bash
python scripts/transformer_vast.py
```
Confirm: loss drops, no OOM, checkpoints appear in `models/vast/`. **If OOM:** lower `batch_size`
(raise `grad_accum_steps` to keep the effective batch). Kill it, restore `max_iters`.

## 4. Full run
```bash
nohup python scripts/transformer_vast.py > train.log 2>&1 &
tail -f train.log
```
Monitor train/val loss (printed every `eval_interval`). The dense-early checkpoint schedule means
you can **kill the run the moment induction appears** in the sweep — don't train to convergence.
Watch the meter against your ceiling.

## 5. Pull checkpoints back to the Mac
```bash
rsync -avz user@<vast-host>:~/mech-interp/models/vast/ ./models/vast/
```
You need the `ckpt_*.pt` series, `model.pt`, and `meta.pt`. Then **stop the Vast instance.**

## 6. The payoff (yours, in the notebooks)
Point a notebook at the vast run — `from transformer_vast import GPTLanguageModel, encode, decode, vocab_size`
(same induction-score + heatmap code as `Interp_2.ipynb`, no other edits; the model classes are
byte-identical, so the checkpoints load directly). Run the induction sweep across `models/vast/ckpt_*.pt`,
watch the induction curve, and find the offset-diagonal head in a repeated-sequence heatmap.

---

### Notes / gotchas
- **`bpe_vocab_size` and `block_size` must match** between `prepare_vast_data.py` and `transformer_vast.py`
  (they're duplicated as plain constants; the prep script writes them into `meta.pt`).
- `prepare_vast_data.py` is idempotent only by file presence — delete `data/vast/` to force a rebuild.
- OOM is the #1 risk. Start at `batch_size=8, grad_accum_steps=8`; raise `batch_size` only with headroom.
- bf16 autocast is on for CUDA automatically (no-op on the Mac, so the file still imports for notebooks).
