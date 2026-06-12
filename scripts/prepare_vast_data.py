"""prepare_vast_data.py — build the tokenized corpus for transformer_vast.py.

Run this FIRST (locally on the Mac is fine — no GPU needed), before launching training:

    python scripts/prepare_vast_data.py --smoke   # ~2MB pipeline test, seconds
    python scripts/prepare_vast_data.py           # the real ~1GB OWT subset

It STREAMS OpenWebText (so it never downloads the full ~40GB — it stops at a token
budget), trains a byte-level BPE tokenizer on a sample, tokenizes the streamed subset to
uint16 data/vast/{train,val}.bin (memmapped by transformer_vast.get_batch), and writes the
tokenizer to models/vast/meta.pt — exactly the {config, tokenizer} shape transformer_bpe.py
uses. At the end it VERIFIES the bins load and a get_batch-style draw returns sane tensors.

This is training infrastructure only; no analysis/interp code lives here.
"""
import os
import argparse
import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── corpus + tokenizer config (keep block_size / vocab in sync with transformer_vast.py) ──
DATASET = 'openwebtext'   # streamed; swap to another HF id only if you also change TEXT_COLUMN
DATASET_SPLIT = 'train'
TEXT_COLUMN = 'text'
bpe_vocab_size = 8192     # must match transformer_vast.py and stay < 65536 (uint16 .bin)
block_size = 512          # informational in meta; training uses transformer_vast.py's value
TOKENIZER_TRAIN_BYTES = 200_000_000  # train BPE on at most this many bytes of streamed text
TARGET_TOKENS = 250_000_000          # ~1GB subset (uint16 -> ~500MB .bin). --smoke shrinks this.
VAL_TOKENS_CAP = 5_000_000           # carve a small val tail (capped; OWT-scale needs little val)

DATA_DIR = os.path.join(ROOT, 'data', 'vast')
CKPT_DIR = os.path.join(ROOT, 'models', 'vast')
TRAIN_BIN = os.path.join(DATA_DIR, 'train.bin')
VAL_BIN = os.path.join(DATA_DIR, 'val.bin')
META_PATH = os.path.join(CKPT_DIR, 'meta.pt')


def stream_corpus():
    """yield raw-text rows from the streamed dataset (never materialises the whole thing)."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=DATASET_SPLIT, streaming=True)
    for row in ds:
        t = row.get(TEXT_COLUMN)
        if t:
            yield t


def train_tokenizer(train_bytes):
    """byte-level BPE (GPT-2 style: every byte representable, no [UNK]), trained on a
    bounded sample of the stream — a tokenizer needs a representative sample, not all of it."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=bpe_vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def sample_iter():
        budget = train_bytes
        for t in stream_corpus():
            yield t
            budget -= len(t.encode('utf-8'))
            if budget <= 0:
                break

    tok.train_from_iterator(sample_iter(), trainer=trainer)
    return tok


def tokenize_to_budget(tok, target_tokens):
    """stream + encode rows until target_tokens reached; return one uint16 array."""
    chunks, total = [], 0
    for t in stream_corpus():
        ids = tok.encode(t).ids
        chunks.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if total >= target_tokens:
            break
        if len(chunks) % 2000 == 0:
            print(f"  tokenized {total:,} / {target_tokens:,} tokens...")
    arr = np.concatenate(chunks)[:target_tokens]
    return arr


def verify():
    """confirm the bins load and a get_batch-style draw is sane (the 'test small' gate)."""
    meta = torch.load(META_PATH)
    tok = Tokenizer.from_str(meta['tokenizer'])
    vocab = tok.get_vocab_size()
    for name, path in [('train', TRAIN_BIN), ('val', VAL_BIN)]:
        data = np.memmap(path, dtype=np.uint16, mode='r')
        assert len(data) > block_size, f"{name} bin too short: {len(data)}"
        assert int(data.max()) < vocab, f"{name} has an id >= vocab_size ({data.max()} >= {vocab})"
        print(f"{name}: {len(data):,} tokens, max id {int(data.max())} (vocab {vocab})")
    # one get_batch-style draw
    data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode='r')
    i = int(np.random.randint(len(data) - block_size))
    x = torch.from_numpy(data[i:i+block_size].astype(np.int64)).unsqueeze(0)
    print("sample batch x shape:", tuple(x.shape))
    print("decoded snippet:", repr(tok.decode(x[0, :40].tolist())))
    print("\nOK — data/vast ready. Next: get the repo + data/vast + venv onto the 4090 (see RUN.md).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='tiny ~2MB run to test the pipeline end-to-end in seconds')
    args = ap.parse_args()

    train_bytes = 5_000_000 if args.smoke else TOKENIZER_TRAIN_BYTES
    target_tokens = 1_000_000 if args.smoke else TARGET_TOKENS
    val_cap = 50_000 if args.smoke else VAL_TOKENS_CAP

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"[1/4] training BPE tokenizer on ~{train_bytes/1e6:.0f}MB of streamed {DATASET}...")
    tok = train_tokenizer(train_bytes)
    torch.save({
        'config': {'vocab_size': tok.get_vocab_size(), 'block_size': block_size},
        'tokenizer': tok.to_str(),
    }, META_PATH)
    print(f"      vocab {tok.get_vocab_size()} -> {META_PATH}")

    print(f"[2/4] tokenizing a ~{target_tokens:,}-token subset (streamed)...")
    arr = tokenize_to_budget(tok, target_tokens)

    print("[3/4] writing train/val bins...")
    n_val = min(val_cap, len(arr) // 100)   # small val tail
    val, train = arr[-n_val:], arr[:-n_val]
    train.tofile(TRAIN_BIN)
    val.tofile(VAL_BIN)
    print(f"      train {len(train):,} | val {len(val):,} tokens")

    print("[4/4] verifying...")
    verify()


if __name__ == '__main__':
    main()
