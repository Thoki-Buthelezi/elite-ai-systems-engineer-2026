#read in the dataset into a variable
with open("datasets/tiny_shakespeare.txt", "r") as f:
    text = f.read()

#tokenize the text into a sorted array
chars = list(set(text))
chars = sorted(chars)
vocab_size = len(chars)

#define hyperparameters for the model
max_iter = 5000
interval_iter = 500
learning_rate = 1e-3
eval_iters = 200
n_embd = 128
dropout = 0.2
head = 8
head_size = n_embd // head



#define a mechanism to map the tokens into numerical IDs
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#define encoding and decoding mechanism
encode = lambda s: [stoi[c] for c in s] #give a  tring return a list of IDs
decode = lambda l: "".join([itos[c] for c in l]) #give a list of IDs return a string

#encode text into data tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)

#separate data into trainig and validation by 90/10
n = int(.9 * len(data))

train_data = data[:n]
val_data = data[n:]

#define block_size which is the size of the chunk of sequence the model will process at a time (Columns)
block_size = 16
#define batch_size which is the number of block size the model will proces (Rows)
batch_size = 16

torch.manual_seed(2002)

#def method to return a training batch and validation batch
def get_batch(split):
    data = train_data if split == "train" else val_data
    #return a 1D vector of random numerical IDs for tokens
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #randomly sammple 16, 16 numerical IDs to use as input
    x = torch.stack([data[i:i+block_size] for i in ix])
    #sample a corresponding target output
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

import torch.nn as nn
from torch.nn import functional as F

"""
    Transformer is a neural network that makes use of attention instead of recurrent and convolution.
    The transformer architecture is built by stacking transformer blocks each consisting of:
    1. Multi-Head attention --> each head attends a different relationship of the sequence
    2. FeedForward network --> to propagate information to later layers
    3. Skip Connections --> I don't know about this one, lol :[]
    4. Layer Normalisation --> to stabilise training

    - it also includes positional encoding to give information about the token's info in the sequence
    - casual masking to preven the token's from seeing future token
"""

#define a single attention lens
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) #B,T, head_size
        k = self.key(x) #B,T, head_size
        v = self.value(x) #B,T, head_size

        #compute scaled dot product
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5 #B,T,head_size * B,head_size,T --> B,T,T
        #apply casual mask to prevent tokens from seeing the future
        wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf"))
        #apply softmax to normalise tokens
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #aggregated the weighted sums with the values
        out = wei @ v #B,T,T * B,T,head_size --> B,T,head_size
        return out
        
#define multi attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, head_size, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concatinate the outputs for each head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

#neural network for feedfoward inside the transformer block
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

#define a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.attention = MultiHeadAttention(n_embd, head_size, n_heads)
        self.ffwd = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #apply pre normalisation to stabilise training and skip connections
        #note that the 'Attention is all you need' paper post normalisation is used
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.ffwd(self.layer_norm2(x))
        return x



class BiLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #define an embedding table that turns each integer token into a vector
        self.token_embbeding_table = nn.Embedding(vocab_size, n_embd) 
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_heads=head) for _ in range(4)],
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    
    def forward(self, idx, target=None):
        B,T = idx.shape
        #look at the logits for the next token
        token_emb = self.token_embbeding_table(idx)
        pos_emb = self.positional_encoding(torch.arange(T, device=idx.device))
        pos_emb = pos_emb.unsqueeze(0)  # (1, T, C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_nex_tokens):
        for _ in range(max_nex_tokens):
            idx_cond = idx[:,-block_size:]
            #make predictions
            logits, loss = self(idx_cond)
            #focus on the last sequence of the logits
            logits = logits[:, -1, :]
            #compute probabilities using softmax
            prob = F.softmax(logits, dim=-1)
            #sample the next logit from multinomial distribution using the probabilities
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

#training loop
def train_model():
    for iter in range(max_iter):
        xb, yb = get_batch("train")
        #make predictions and compute the loss
        logits, loss = model(xb, yb)

        #initialise weights gradients to zero for each iteration
        optimizer.zero_grad(set_to_none=True)

        #perform backprop
        loss.backward()

        #update model parameters
        optimizer.step()

        #display the loss in time intervals
        if iter % interval_iter == 0:
            losses = estimate_loss()
            print(f"steps:{iter} | train loss:{losses['train']:.4f} | val loss:{losses['val']:.4f}")



model = BiLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_model()

# logits, loss = model(xb, yb)
# print("logits")
# print(logits.shape)
# print(logits)
# print("loss")
# print(loss.item())

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_nex_tokens=1000)[0].tolist()))

