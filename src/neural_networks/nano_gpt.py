"""
    This is my annoted version of Andrej Karpathy nanoGPT on tinyshakespeare.
    My implementation follows his implementation except I code it without 
    looking and describing every line how i understand it
"""

#import libriries to used
import torch 
import torch.nn as nn
from torch.nn import functional as F

#set seed to 1337 to reproduce same results as Karpathy
torch.manual_seed(1337)

#define and declare hyperparamters to use to build the model
#use same values as Karpathy
block_size = 64
batch_size = 32 
learning_rate = 1e-4
max_iters = 5000
eval_iters = 200
eval_interval = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embd = 128
num_heads = 4
head_size = n_embd // num_heads
dropout = 0.2



#read in shakespeare text file into a variable text
with open("datasets/tiny_shakespeare.txt", "r") as f:
    text = f.read()


#tokenizer text into subword/units
chars = list(set(text))
#sort 
chars = sorted(chars)
vocab_size = len(chars)

#define enconding/decoding mechanism
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#convert each subword into a numerical id
encode = lambda s: [stoi[c] for c in s]
#convert each numerical id into a subword
decode = lambda l: "".join([itos[c] for c in l])

#encode the text data into torch tensor
data = torch.tensor(encode(text), dtype=torch.long)

#split the data into test data and validation data

n = int(.9 * len(data))

# and 90% train
train_data = data[:n]
#10% validate
val_data = data[n:]

#define a function to sample a batch
def get_batch(split):
    data = train_data if split=="train" else val_data
    #create a random 1D vector of 32 integers to sample 32 chunks of sequnce with length 8 each
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #stack up each chunk of sequence int a tensor of shape (32 * 8)
    x = torch.stack([data[i:i+block_size] for i in ix])
    #stack up a corresponding output for each context
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

#function to estimate the loss on train and val data
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    #tell the model not to store intermediate value because we are not going to use backprop
    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                X, Y = X.to(device), Y.to(device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

#define the feedfoward neural network to use in the transformer block
class Feedforward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    
    def forward(self, x):
      return self.net(x)    


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)          
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) #(B,T,head_size)
        q = self.query(x) #(B,T,head_size)
        v = self.value(x) #(B,T,head_size)

        #compute attention scores "affinities"
        #scale to prevent vanishing gradients
        wei = q @ k.transpose(-2,-1) * head_size**-0.5 #(B,T,head_size) @ (B, head_size, T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf")) #mask out future positions
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)

        #perform the weighted aggregation of the values
        out = wei @ v #(B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
                

class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_head = MultiHeadAttention(n_heads, head_size)
        self.ffwd = Feedforward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    
    def forward(self, x):             
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

#now feed the data into the neural netowk
class BiLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=num_heads) for _ in range(4)], nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) #(B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B*T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #crop idx to the last block_size tokens
            #make predictions
            logits, loss = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (B* C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)    
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled index to the running sequence             
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx
#instantiate the model
model = BiLanguageModel()  
m = model.to(device)

#construct an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train():
    for iter in range(max_iters):
        #every once in a while evaluate the loss on train data and var data
        if iter % eval_interval == 0:
            losses = estimate_loss(model=model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        #sample a batch of data
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)
        #evaluate the loss
        logits, loss = model(xb, yb)
        #set gradients of parameters to zero
        optimizer.zero_grad(set_to_none=True)
        #perform backpropagation
        loss.backward()
        #update paramertes
        optimizer.step()

if __name__ == "__main__":
    #train the model
    train()
    
    #generate from model
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

    #save the pretrained model
    torch.save(model.state_dict(), "nano_gpt_model.pt")












