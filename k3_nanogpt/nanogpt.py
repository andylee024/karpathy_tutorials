import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
vocab_size = 65
n_embd = 32
head_size = n_embd
num_heads = 4

max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 200

# prepare the dataset
with open('shakespeare_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# build a vocabulary
chars = sorted(list(set(text)))

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# setup data
data = torch.tensor(encode(text), dtype=torch.long)

# split data as training & validation 
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]


def get_batch(split, batch_size=batch_size, block_size=block_size):
    data = train_data if split == 'train' else validation_data

    # generate batch indices
    last_valid_index = len(data) - block_size
    batch_indices = torch.randint(0, last_valid_index, (batch_size,)) 

    # query dataset to generate batch data
    x = torch.stack([data[idx: idx+block_size] for idx in batch_indices], dim=0)
    y = torch.stack([data[idx+1: idx+block_size+1] for idx in batch_indices], dim=0)
    
    return x,y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    
    model.train()
    return out


class AttentionHead(nn.Module):
    """One head of self-attention """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # NN matrices
        self.head_size = head_size
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape
        
        # setup attention variables
        kx = self.key(x) # output is (B, T, H) 
        qx = self.query(x) # (B, T, H)
        vx = self.value(x) # (B, T, H)
        triangle = torch.tril(torch.ones(T, T)) 

        # compute raw attention scores , dot(K, Q)
        # attention scores are computed pair-wise (s_ij = key(s_i) * query(s_j))
        kxT = kx.transpose(dim0=-2, dim1=-1) # (B, T, H) -> (B, H, T) 
        attention_values = qx @ kxT # (B, T, H) * (B, H, T) = (B, T, T) 
        normalization_constant = self.head_size ** -0.5

        # compute attention scores (aka affinities)
        # attention only looks forward for language modeling
        attention_values = attention_values * normalization_constant # (B, T, T)
        attention_values = attention_values.masked_fill(triangle == 0, float('-inf')) # (B, T, T), 
        attention_weights = F.softmax(attention_values, dim=-1) # (B, T, T)

        # compute weighted aggregation of attention scores and values
        v = self.value(x)  # (B, T, H)
        out = attention_weights @ v # (B, T, T) * (B, T, H) = (B, T, H)

        return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        """Multiheaded attention implementation

        :param num_heads: total number of heads in multiheaded attention block
        :param head_size: size of each head

        The output is a concatenation of all single attention blocks
        :return : B x T x (num_heads * head_size)
        """
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        # each h(x) is a single attention head
        # h(x) output is (B, T, H)
        return torch.cat([h(x) for h in self.heads], dim=-1) # since we're stacking on dim=-1, we want this value to equal H
    

class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), 
                                 nn.ReLU())
    
    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    """Current implementation: single-head attention bigram language model"""

    def __init__(self):
        
        super().__init__()

        B = batch_size # 32
        T = block_size # 8
        C = vocab_size # 65
        H = n_embd # 32
        h = num_heads # 4

        # Embedding matrices 
        self.token_embedding_table = nn.Embedding(C, H) # in: (B, T, C), out: (B, T, H)
        self.position_embedding_table = nn.Embedding(T, H) # in : (B, T, T), out: (B, T, H)

        # Transformations

        # V1 - single attention
        # self.sa_head = AttentionHead(num_heads=4, head_size=n_embd) # V1

        # V2 - multiheaded attention
        self.sa_head = MultiHeadedAttention(num_heads=4, head_size=n_embd//4) # in: (B, T, H) , out: (B, T, H)
        self.ffwd = FeedForward(H, H) # in: (B, T, H) , out: (B, T, H) 
        self.lm_head = nn.Linear(H, C) # in: (B, T, H), out: (B, T, C)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        C = vocab_size

        # embed x
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb

        # transform x 
        x = self.sa_head(x) 
        x = self.ffwd(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            logits_reshape = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_reshape, targets)
        
        return logits, loss
     
    def generate(self, idx, max_new_tokens):
        """Generate a new sequence of characters for each mini-batch-example.
        
        :param idx: represents sequences in mini-batch (B, T, C) tensor
        - B - batch size
        - T - sequence length
        - C - number of dimensions in embedding space (aka channels)

        :param max_new_tokens: the number of new characters we append to each existing sequence
        :return:
        - (B, T, C + max_new_tokens) tensor, representing new generated sequences
        """
        B, T = idx.shape
        
        for _ in range(max_new_tokens):

            # crop input_sequence to the last block_size tokens
            # if input_sequence is longer than block_size, positional embedding breaks
            idx_cond = idx[:, -block_size:]

            # forward pass generates unnormalized probabilities 
            logits, _ = self.forward(idx_cond, targets=None) # (B, T, C)

            # get probability distribution for the last character in the sequence to predict next character
            p = logits[:, -1, :] # (B, 1, C)

            # normalize probabilities so we can sample
            softmax_p = F.softmax(p, dim=1) # expected (B x C tensor)
            
            # sample next character
            next_char = torch.multinomial(softmax_p, num_samples=1) # (B x 1 tensor)

            # append next character
            idx = torch.cat((idx, next_char), dim=1) # (B x T+1 x C)
        
        return idx


def run_training(model):

    # inspection 
    # print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') # print number of parameters in model

    # setup 
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # run training loop
    for iter in range(max_iters + 1):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model
    

def sample_from_model(model):
    
    # pad initial input so model initialization works
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)

    # sample from model
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

    # write output to text file
    # open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

if __name__ == "__main__":
    model = BigramLanguageModel()
    model = run_training(model)
    sample_from_model(model)

