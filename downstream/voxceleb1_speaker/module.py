import torch
from torch import nn
 

class MP(nn.Module):
    """
    Mean Pooling
    """
    def __init__(self, input_dim, out_dim):
        super(MP, self).__init__()
        self.linear_layer = nn.Linear(input_dim, out_dim)
    def forward(self, x, length=None):
        """
        x:  B x L x H
        """
        hidden = self.linear_layer(x)
        avg_list = []
        for i in range(len(x)):
            avg_vector = torch.mean(hidden[i,:length[i]],dim=0)
            avg_list.append(avg_vector)
        output =torch.stack(avg_list)
        return output, None

class AP(nn.Module):
    """
    Attentive Pooling
    """
    def __init__(self, input_dim, out_dim):
        super(AP, self).__init__()
        self.linear_layer == nn.Linear(input_dim, out_dim)
        self.q_layer = nn.Linear(out_dim, 1)
        self.act_fn = nn.Tanh()
        self.normalized = nn.Softmax(dim=1)
    def forward(self, x, length=None):
        """
        x:  B x L x H
        att_mask: B x L x 1
        """
        hidden = self.act_fn(self.linear_layer(x))
        logits = self.q_layer(hidden)
        att_mask = [torch.lt(torch.arange(x.size(1)), torch.ones_like(x[i]).fill_(length[i])) for i in len(length)]
        
        if att_mask is not None:
            logits  = logits + att_mask
        
        att_weights = self.normalized(logits)
        output = torch.sum(att_weights * logits, dim=1)

        return output, att_weights


class Model(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, agg_method, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.agg_module = eval(agg_method)(self.input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
    def forward(self, x, length):
        hidden,_ = self.agg_module(x, length)

        spk_num_out = self.linear(hidden)

        return spk_num_out
    