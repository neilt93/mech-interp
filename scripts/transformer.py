import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F # stateless functions from nn, no stored params

# repo root = parent of scripts/, so data + checkpoint paths resolve no matter the cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# hyperparameters
batch_size = 64 # parallel sequences processed per step
block_size = 256 # max length input context
max_iters = 5000 # total training steps
eval_interval = 500 # how often to print train/val loss (fewer pauses; each eval is pricey now)
checkpoint_interval = 250 # steady-state snapshot cadence after the dense early phase
# dense-early snapshot schedule: circuits form fast and this run is one-shot (seed=None),
# so capture the early phase change at high resolution, then coast at checkpoint_interval.
checkpoint_steps = {0, 10, 25, 50, 100, 150, 200, 350, 500, 750}
learning_rate = 3e-4 # AdamW step size (smaller again now the model is much bigger)
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # mac specific
eval_iters = 200 # batches averaged per loss estimate (smooths the reading)
n_embd = 384 # embedding dimension,the model's width
n_head = 6 # number of attention heads per block (384/6 = 64 per head)
n_layer = 6 # number of stacked transformer blocks (depth)
dropout = 0.2 # fraction of activations randomly zeroed for regularisation (on now - model can overfit)
gen_tokens = 2000 # number of tokens to generate from the trained model at the end
seed = None

# torch.manual_seed(1337)
if seed is not None:
    torch.manual_seed(seed)

with open(os.path.join(ROOT, 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    # y is one ahead of x to be used as prediction target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# estimate loss
@torch.no_grad() # don't waste memory tracking gradients
def estimate_loss():
    out = {}
    model.eval() # set to eval mode, do not dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # stores a tensor as part of the module but not a trainable param

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # stick the outputs of each head side by side 
        out = self.dropout(self.proj(out)) # W_o reblends the rows, adam trains this too, then dropout
        return out
    
# this is the perceptron
class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity (an MLP).
    After attention gathers context from other tokens, this processes the gathered 
    information. ReLU is the model's only nonlinearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# One decoder block
class Block(nn.Module):
    """ Transformer block: communication followed by computation 
    This is a decoder (parallel attention heads followed by a perceptron post merging)"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        # LayerNorm: normalises each token's features to stabilise training
        self.ln1 = nn.LayerNorm(n_embd) # normalises the input to attention (pre-norm)
        self.ln2 = nn.LayerNorm(n_embd) # normalises the input to the feed-forward (pre-norm)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # what attention gathered
        x = x + self.ffwd(self.ln2(x)) # what the feed-forward computed
        return x
    
# the whole thing
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token index looks up its n_embd embedding vector (NOT logits - lm_head produces those)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# run it all — guarded so `import transformer` only loads the classes/config above,
# not this 39-minute training run. notebooks can `from transformer import GPTLanguageModel`.
if __name__ == '__main__':
    model = GPTLanguageModel()
    m = model.to(device)
    print("Device:", device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # developmental-interp checkpoints: snapshot weights through training so we can
    # load the series back later and watch circuits form. models/ is gitignored (weights are large).
    ckpt_dir = os.path.join(ROOT, 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    # save run config + vocab once so any snapshot can be rebuilt and decoded without this file
    torch.save({
        'config': {
            'vocab_size': vocab_size, 'block_size': block_size, 'n_embd': n_embd,
            'n_head': n_head, 'n_layer': n_layer, 'dropout': dropout,
        },
        'stoi': stoi, 'itos': itos,
    }, os.path.join(ckpt_dir, 'meta.pt'))

    train_start = time.time()      # total training wall-clock
    interval_start = train_start   # resets each eval interval to measure throughput
    best_val_loss = float('inf')   # track the best val loss so we only checkpoint improvements
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            now = time.time()
            # ms per step since the last eval print (excludes the eval itself on the first iter)
            steps_done = eval_interval if iter > 0 else 1
            ms_per_step = (now - interval_start) / steps_done * 1000
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} "
                  f"({ms_per_step:.1f} ms/step, {now - train_start:.1f}s elapsed)")
            interval_start = now
            # best-val checkpointing: only save when val improves, so model.pt holds
            # the best weights rather than the overfit final ones.
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pt'))
                print(f"  new best val loss {best_val_loss:.4f} -> saved models/model.pt")

        # developmental-interp snapshot: dense early (checkpoint_steps), then every
        # checkpoint_interval, plus the final step. independent of best-val. iter padded for sort order.
        if iter in checkpoint_steps or iter % checkpoint_interval == 0 or iter == max_iters - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f'ckpt_{iter:06d}.pt'))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    train_time = time.time() - train_start
    print(f"\nTraining done: {max_iters} steps in {train_time:.1f}s ({train_time / max_iters * 1000:.1f} ms/step avg)")
    print(f"Best val loss: {best_val_loss:.4f} (weights saved to models/model.pt)")

    # reload the best-val checkpoint so we generate from the best weights, not the final overfit ones
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'model.pt')))
    model.eval()

    # generate from the model
    gen_start = time.time()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=gen_tokens)[0].tolist()))
    print(f"\nGenerated {gen_tokens} tokens in {time.time() - gen_start:.1f}s")