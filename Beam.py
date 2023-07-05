import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    #outputs就是代替训练时候 target输入decoder
    outputs = torch.LongTensor([[init_tok]], device=opt.device)

    trg_mask = nopeak_mask(1, opt)
    
    out = model.out(model.decoder(outputs,e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len, device=opt.device).long()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1), device=opt.device)
    e_outputs[:, :] = e_output[0]
    # outputs就是代替训练时候 target输入decoder
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    '''集束搜索beam search
    每次保留概率最高的三个词，然后三个词再预测出9个词；在9个词中保留概率最大的三个词，以及三个词所对应的前半句（其余的就删除了）；
    beam_size=3在所有时间步，保留3个概率最高词；top_beams=2最终会保留2个翻译结果；
    相对的：贪婪解码greedy decoding：每次保留概率最大的词语；
    '''
    row = k_ix // k#行列号
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']#eos是句子开始
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)#pad是补全
    ind = None
    for i in range(2, opt.max_len):#从第二个到maxlen=80 （第一个是开始符）
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i],###每次decoder输入当前预测的前一个单词
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
