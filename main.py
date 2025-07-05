
import requests
from bs4 import BeautifulSoup
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import zipfile
import urllib.request
from tqdm.notebook import tqdm


def load_cornell_movie_dialogs_words():
    dataset_url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_file_name = "cornell_movie_dialogs_corpus.zip"
    extracted_folder_name = "cornell movie-dialogs corpus"
    dialogs_file_name = os.path.join(extracted_folder_name, "movie_lines.txt")

    if not os.path.exists(zip_file_name):
        urllib.request.urlretrieve(dataset_url, zip_file_name)
        print('download is finished')
    else:
      pass
    if not os.path.exists(extracted_folder_name):
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
      pass

    if not os.path.exists(dialogs_file_name):
        print(f"No dialog file {dialogs_file_name}")
        return ""
    lines = []
    with open(dialogs_file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' +++$+++ ')
            if len(parts) >= 5:
                clean_text = parts[4].lower()
                clean_text = clean_text.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ')
                clean_text = clean_text.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ')
                clean_text = clean_text.replace(':', ' ').replace(';', ' ').replace('"', ' ').replace("'", ' ')
                clean_text = clean_text.replace('-', ' ')
                lines.append(clean_text)
    text = "".join(lines)
    print(f"len of dialogs: {len(lines)} len of whole text: {len(text)} ")

    max_test_chars = 100000
    if len(text) > max_test_chars:#cutting text
        text = text[:max_test_chars]

    return text

text = load_cornell_movie_dialogs_words()









###########################################################################################################


import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
with open("corpus.txt","w",encoding="utf-8") as f:
  f.write(text)

tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
trainer = BpeTrainer(special_tokens=['[UNK]',"[CLS]","[SEP]"],vocab_size=5000 )

tokenizer.train(files = ['corpus.txt'],trainer = trainer)

vocab = tokenizer.get_vocab()

embedding = nn.Embedding(len(vocab), embedding_dim= 8 )



data = torch.tensor((tokenizer.encode(text)).ids,dtype=torch.long)[None, :]
print(data.shape, 'DATA SHAPE')


# xs2 =  embedding(data)
vocab_size = len(vocab)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class FeedForward(nn.Module):
  def __init__(self,embed_dim,hidden_dim):
    super().__init__()
    self.fc1 = nn.Linear(embed_dim,hidden_dim)
    self.fc2 = nn.Linear(hidden_dim,embed_dim)
  def forward(self,x):
    A1 = F.relu(self.fc1(x))
    A2 = self.fc2(A1)
    return A2


class MultiHeadAttention(nn.Module):
  def __init__(self,embed_dim, num_heads,seq_len = 4):
    super().__init__()
    self.qkv = nn.Linear(embed_dim, embed_dim * 3)
    self.out = nn.Linear(embed_dim,embed_dim)
    self.head_dim = torch.tensor(embed_dim // num_heads)
    self.num_heads = num_heads
    self.seq_len = seq_len
    self.embed_dim = embed_dim
  def forward(self,x):
    B,T,C = x.shape
    qkv = self.qkv(x)

    q, k ,v = qkv.chunk(3,dim=-1)
    q = q.reshape(B,T,self.num_heads, self.head_dim).permute(0,2,1,3)
    k = k.reshape(B,T,self.num_heads, self.head_dim).permute(0,2,1,3)
    v = v.reshape(B,T,self.num_heads, self.head_dim).permute(0,2,1,3)

    scores= torch.matmul(q,k.permute(0,1,3,2)) / torch.sqrt(self.head_dim)

    mask = torch.tril(torch.ones(T,T,device = x.device)).unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores,dim=-1)
    attention_output = torch.matmul(attention_weights,v)
    attention_output = attention_output.permute(0,2,1,3).reshape(B,T,self.embed_dim)

    output = self.out(attention_output)
    return output



class multiheadgpt(nn.Module):
  def __init__(self,embed_dim,hidden_dim,num_heads,max_len = len(text)):
    super().__init__()
    self.embed_dim = embed_dim
    self.fnn = FeedForward(embed_dim,hidden_dim)
    self.attn = MultiHeadAttention(embed_dim,num_heads)
    self.token_embedding = nn.Embedding(vocab_size,embed_dim)
    self.pos_embedding = nn.Embedding(max_len,embed_dim)
    self.o_p = nn.Linear(embed_dim,vocab_size)

    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
  def forward(self,idx):
    B,T = idx.shape

    token_emb = self.token_embedding(idx)
    pos = torch.arange(T,device = idx.device).unsqueeze(0)
    pos_emb = self.pos_embedding(pos)

    x = token_emb + pos_emb
    x_attn = self.attn(self.norm1(x))
    x = x+ x_attn
    x_fnn = self.fnn(self.norm2(x))
    x = x + x_fnn
    logits = self.o_p(x)
    return logits




data.to(device)
model = multiheadgpt(embed_dim=256,hidden_dim=512,num_heads=2).to(device)
model.train()
epochs = 200
optimizier = torch.optim.AdamW(model.parameters(), lr =  1e-4)
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
from tqdm import tqdm


for epoch in tqdm(range(epochs)):
  x = data[:,:-1].to(device)
  y = data[:,1:].to(device)

  optimizier.zero_grad()
  pred = model(x)
  loss = loss_fn(pred.view(-1,vocab_size),y.view(-1))
  loss.backward()
  optimizier.step()
  if epoch % 20 == 0:
    print()
    print(f"Loss: {loss.item()}")




@torch.no_grad()
def generate(model,start_text,temperature,top_k,max_new_tokens=40):
    model.eval()
    tokens = tokenizer.encode(start_text).ids
    x= torch.tensor(tokens, dtype=torch.long)[None,:]
    x =x.to(device)
    for _ in range(max_new_tokens):
        x_cond = x[:, -128:]
        logits = model(x_cond)
        logits = logits[:,-1,:]
        logits = logits / temperature
        values,indices = torch.topk(logits,top_k)
        probs = F.softmax(values, dim=-1)

        next_token = indices.gather(-1,  torch.multinomial(probs,num_samples=1) )
        x= torch.cat([x,next_token], dim=1)
    return tokenizer.decode(x[0].tolist())

generate(model,'the',temperature=1,top_k=4)
