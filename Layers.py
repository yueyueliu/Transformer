import torch
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm

##一个encoder block的网络
class EncoderLayer(nn.Module):
    # d_model初始化向量维度,
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model) #norm
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)# multihead attention
        self.ff = FeedForward(d_model, dropout=dropout) #feed forward
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)#（x，y1 ，512）初始词embedding
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))#add & att(norm)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))#add & ff(norm)
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    # d_model初始化向量维度,
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model) #norm
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)# mask multihead attention
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)# multihead attention
        self.ff = FeedForward(d_model, dropout=dropout) #feed forward

    #src_mask.(x,1,y1) trg_mask.(x,y2,y2)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))#add & maskatt(norm)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))#add & att(norm)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))#add & ff(norm)
        return x