import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Projector(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, fc_hid_dim, embeddings=None, device="cuda"):
        super(Projector, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embed.weight.data.copy_(torch.from_numpy(np.array(embeddings)))
            self.embed.weight.requires_grad = True
        self.drop = nn.Dropout(0.2)
        self.rnn = RNN(emb_dim, hid_dim, n_layers)

#         self.reshape = nn.Sequential(nn.Linear(hid_dim*2*n_layers, emb_dim*2), nn.ReLU())
        self.linear = nn.Sequential(# nn.Linear(emb_dim*n_layers, fc_hid_dim),
                                    nn.Linear(hid_dim*2*n_layers, fc_hid_dim),
                                    nn.ReLU(),
#                                     Maxout(fc_hid_dim, fc_hid_dim, 3),
                                    nn.Linear(fc_hid_dim, fc_hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(fc_hid_dim, fc_hid_dim),
                                    nn.ReLU(),
                                    nn.Linear(fc_hid_dim, 4))
        
        
    
    def forward(self, data, ind_sort, ind_unsort, length):
        
        prem = self.drop(self.embed(data[0]))
        hyp = self.drop(self.embed(data[1]))
        
        fa = self.rnn((prem, hyp), ind_sort, ind_unsort, length) # (b, hid*2)
        out = self.linear(fa)
        
        return out

    
class RNN(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers):
        super(RNN, self).__init__()
        
        self.n_layers, self.hid_dim = n_layers, hid_dim

#         self.cnn = nn.Sequential(nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=1, padding=int(3/2)),
#                                  nn.ReLU(),
# #                                  nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=1, padding=int(3/2)),
# #                                  nn.ReLU()
#                                 )
        self.rnn_prem = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=False)
#         self.rnn_hyp = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, bidirectional=False)
        self.rnn_hyp = self.rnn_prem

#         self.bn = nn.BatchNorm1d(hid_dim*2*n_layers)
        
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * 1, batch_size, self.hid_dim).cuda()
        cell = torch.zeros(self.n_layers * 1, batch_size, self.hid_dim).cuda()
        return hidden, cell

    def forward(self, x, ind_sort, ind_unsort, lengths):
        batch_size = x[0].size(0)
        hidden, cell = self.init_hidden(batch_size)
        
#         prem = prem.transpose(1,2)
#         prem = self.cnn(prem) # batch_size * hidden_size * seq_len
#         prem = prem.transpose(1,2)
#         hyp = hyp.transpose(1,2)
#         hyp = self.cnn(hyp)
#         hyp = hyp.transpose(1,2)    
        
#         prem = x[0].index_select(0, ind_sort[0])
#         prem_len = lengths[0].index_select(0, ind_sort[0]).cpu().numpy()
#         prem = torch.nn.utils.rnn.pack_padded_sequence(prem, prem_len, batch_first=True)
#         _, (prem, _) = self.rnn_prem(prem, (hidden, cell)) # (n_layers, b, h)
#         prem = prem.index_select(1, ind_unsort[0])
#         prem = prem.transpose(0, 1).contiguous().view((batch_size, 1, -1)).contiguous()
        
#         hyp = torch.cat([prem.expand_as(x[1]), x[1]], dim=2)
#         hyp = hyp.index_select(0, ind_sort[1])
#         hyp_len = lengths[1].index_select(0, ind_sort[1]).cpu().numpy()
                
#         hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_len, batch_first=True)
        
#         _, (hyp, _) = self.rnn_hyp(hyp, (hidden, cell))
#         hyp = hyp.index_select(1, ind_unsort[1])        
#         hyp = hyp.transpose(0, 1).contiguous().view((batch_size, -1)).contiguous()
#         out = torch.cat((prem[:, 0, :], hyp), dim=1)


        prem = x[0].index_select(0, ind_sort[0])
        hyp = x[1].index_select(0, ind_sort[1])
        prem_len = lengths[0].index_select(0, ind_sort[0]).cpu().numpy()
        hyp_len = lengths[1].index_select(0, ind_sort[1]).cpu().numpy()
        prem = torch.nn.utils.rnn.pack_padded_sequence(prem, prem_len, batch_first=True)
        hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp, hyp_len, batch_first=True)
        _, (prem, _) = self.rnn_prem(prem, (hidden, cell)) # (n_layers, b, h)
        hidden, cell = self.init_hidden(batch_size)
        _, (hyp, _) = self.rnn_hyp(hyp, (hidden, cell))
        prem = prem.index_select(1, ind_unsort[0])
        hyp = hyp.index_select(1, ind_unsort[1])  
        prem = prem.transpose(0, 1).contiguous().view((batch_size, -1)).contiguous()
        hyp = hyp.transpose(0, 1).contiguous().view((batch_size, -1)).contiguous()
        out = torch.cat([prem, hyp], dim=1)

        
        return out
        
        
class BagOfWords(nn.Module):
    def __init__(self, emb_dim):
        super(BagOfWords, self).__init__()
        self.emb_dim = emb_dim
        self.bn = nn.BatchNorm1d(emb_dim*2)
    
    def avg_emb(self, data, length):
        
        out = torch.sum(data, dim=1) # (b, emb)
        out /= length.float().unsqueeze(1)
        return out
        
    def forward(self, data, length):
        
        out0 = self.avg_emb(data[0], length[0])
        out1 = self.avg_emb(data[1], length[1])

        out = torch.cat((out0, out1), dim=1)
        out = self.bn(out)
        return out
    
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    query: (bsz, 1, d)
    key: (bsz, k, d)
    value: (bsz, k, d)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k) # (bsz, 1, k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
   
    
class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m