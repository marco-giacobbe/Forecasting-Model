import torch
from torch import nn
import math
from pykan import KAN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).view(max_len, 1)
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.view(1, max_len, d_model).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, kan=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "nhead must be multiple of d_model"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        if kan:
            self.w_q = KAN([d_model, d_model])
            self.w_k = KAN([d_model, d_model])
            self.w_v = KAN([d_model, d_model])
            self.w_o = KAN([d_model, d_model])
        else:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def split_heads(self, x):
        seq_len, batch_size, _ = x.size()
        x = x.view(seq_len, batch_size * self.nhead, self.d_k)
        x = x.transpose(0, 1)
        x = x.view(batch_size, self.nhead, seq_len, self.d_k)
        return x

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.permute(2, 0, 1, 3).contiguous().view(batch_size * seq_len, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask):
        scale_factor = 1 / math.sqrt(self.d_k)
        attention_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
        attention_weight += mask
        attention_weight = self.drop(torch.softmax(attention_weight, dim=-1))
        return torch.matmul(attention_weight, v)

    def forward(self, q, k, v, mask):
        seq_len, batch_size, _ = q.size()
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attention_output = self.scaled_dot_product_attention(q, k, v, mask)
        attention_output = self.combine_heads(attention_output)
        return self.w_o(attention_output).view(seq_len, batch_size, self.d_model)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, ff, dropout, kan=False):
        super(FeedForward, self).__init__()
        if kan:
            self.linear1 = KAN([d_model, ff])
            self.linear2 = KAN([ff, d_model])
        else:
            self.linear1 = nn.Linear(d_model, ff)
            self.linear2 = nn.Linear(ff, d_model)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.drop(self.relu(self.linear1(x))))
    
class EncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, ff, dropout, kan=False):
    super(EncoderLayer, self).__init__()
    self.mha_block = MultiHeadAttention(d_model, nhead, dropout, kan=kan)
    self.ff_block = FeedForward(d_model, ff, dropout, kan=kan)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.drop = nn.Dropout(dropout)

  def forward(self, x, mask):
    mha_out = self.drop(self.mha_block(x, x, x, mask))
    x = self.norm1(x + mha_out)
    ff_out = self.drop(self.ff_block(x))
    x = self.norm2(x + ff_out)
    return x
  
class LTSTransformer(nn.Module):
  def __init__(self, d_model, nhead, ff, nlayer, dropout=0.1, kan=False, finetuning=False):
    super(LTSTransformer, self).__init__()
    self.pos_encoder = PositionalEncoding(d_model, max_len=300)
    self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, ff, dropout) for _ in range(nlayer)])
    self.decoder = KAN([d_model, 1]) if kan else nn.Linear(d_model, 1)

    # for finetuning we freeze all decoder weights
    if finetuning:
      for param in self.decoder.parameters():
        param.requires_grad = False

  def generate_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


  def forward(self, x):
    self.mask = self.generate_mask(x.shape[0]).to(x.device)

    x = self.pos_encoder(x)
    for layer in self.encoder:
      x = layer(x, self.mask)
    return self.decoder(x)