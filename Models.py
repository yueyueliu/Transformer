import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    #d_model初始化向量维度, N block个数,
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N#6个encoder block
        self.embed = Embedder(vocab_size, d_model)#初始化词向量#（13724，512）
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)################一个encoder block的实现
        self.norm = Norm(d_model)

    # src（x,y1）,src_mask(x,1,y1),
    def forward(self, src, mask):
        x = self.embed(src)#（x，y1 ，512）初始词embedding
        x = self.pe(x)#（x，y1 ，512） 加入位置信息
        for i in range(self.N): #block个数=层数
            x = self.layers[i](x, mask)###6个block
        return self.norm(x)  #encoder的输出(编码矩阵C)
    
class Decoder(nn.Module):
    # d_model初始化向量维度, N block个数,
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)#初始化词向量#（23469，512）
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)################一个dncoder block的实现
        self.norm = Norm(d_model)
    #decoder接收 目标词向量；encoder的输出(编码矩阵C)；mask
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)#（x，y2 ，512）初始词embedding
        x = self.pe(x)#（x，y2 ，512） 加入位置信息
        for i in range(self.N):#block个数=层数
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    # src（x,y1）, trg(x,y2), src_mask(x,1,y1), trg_mask(x,y2,y2)
    def forward(self, src, trg, src_mask, trg_mask):
        #encoder直接收输入词向量
        e_outputs = self.encoder(src, src_mask)
        print("DECODER")
        #decoder接收 目标词向量；encoder的输出(编码矩阵C)；mask
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    model.to(opt.device)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model
    
