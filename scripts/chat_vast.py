"""chat_vast.py — interactively generate from the trained Vast (GPT-2-small) model.

    python scripts/chat_vast.py                 # best-val model.pt, 150 new tokens
    python scripts/chat_vast.py -n 300          # longer continuations
    python scripts/chat_vast.py --ckpt models/vast/ckpt_000750.pt   # any checkpoint

It's a BASE completion model (OpenWebText, not instruction-tuned): give it the start
of something (a headline, a sentence opener) and it continues the text — it doesn't
answer questions. Empty line or Ctrl-D / Ctrl-C to quit.
"""
import os
import sys
import argparse
import torch

# resolve repo root so paths work no matter the cwd, and make transformer_vast importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from transformer_vast import GPTLanguageModel, encode, decode, device


def main():
    ap = argparse.ArgumentParser(description='chat with the trained Vast model')
    ap.add_argument('--ckpt', default='models/vast/model.pt', help='checkpoint to load')
    ap.add_argument('-n', '--max-new-tokens', type=int, default=150, help='tokens to generate per prompt')
    args = ap.parse_args()

    # checkpoints were saved on CUDA -> map_location for the Mac
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval().to(device)
    print(f"loaded {args.ckpt} on {device} — base completion model, give it a text opener.")
    print("empty line or Ctrl-C to quit.\n")

    while True:
        try:
            prompt = input('prompt> ')
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not prompt.strip():
            break
        ids = torch.tensor([encode(prompt)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens)[0].tolist()
        print(decode(out) + '\n')


if __name__ == '__main__':
    main()
