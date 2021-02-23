import torch
import torch.nn as nn

class ASP(nn.Module):
    ''' Self Attention Pooling module incoporate attention mask'''

    def __init__(self, out_dim, input_dim):
        super(ASP, self).__init__()

        # Setup
        # self.act_fn = nn.ReLU()
        self.linear = nn.Linear(input_dim, out_dim)
        self.sap_layer = AP(out_dim, out_dim)
    
    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        #Encode
        # feature = self.act_fn(feature)
        feature = self.linear(feature)
        sap_vec, att_w = self.sap_layer(feature, att_mask)
        variance = torch.sqrt(torch.sum(att_w * feature * feature, dim=1) - sap_vec**2 + 1e-6)
        statistic_pooling = torch.cat([sap_vec, variance], dim=-1)

        return statistic_pooling

class AP(nn.Module):
    """
    Implementation of Attentive Pooling 
    """
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(AP, self).__init__()
        self.W_a = nn.Linear(input_dim,hidden_dim)
        self.W = nn.Linear(hidden_dim, 1)
        self.act_fn = nn.Tanh()
        self.softmax = nn.functional.softmax
    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
        att_w : size (N, T, 1)
        
        return:
        utter_rep: size (N, H)
        """
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w

class MP(nn.Module):

    def __init__(self, **kwargs):
        super(MP, self).__init__()

    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        agg_vec_list = []
        for i in range(len(feature)):
            if torch.nonzero(att_mask[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature[i])
            else:
                length = torch.nonzero(att_mask[i] < 0, as_tuple=False)[0] + 1
            agg_vec=torch.mean(feature[i][:length], dim=0)
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)


class SP(nn.Module):
    ''' Statistic Pooling incoporate attention mask'''

    def __init__(self, **kwargs):
        super(SP, self).__init__()

        # Setup
        self.sp_layer = MP()
    
    def forward(self, feature, att_mask):

        ''' 
        Arguments
            feature - [BxTxD]   Acoustic feature with shape 
            att_mask   - [BxTx1]     Attention Mask logits
        '''
        #Encode
        mean_vec = self.sp_layer(feature, att_mask)
        variance_vec_list = []
        for i in range(len(feature)):
            if torch.nonzero(att_mask[i] < 0, as_tuple=False).size(0) == 0:
                length = len(feature[i])
            else:
                length = torch.nonzero(att_mask[i] < 0, as_tuple=False)[0] + 1
            variances = torch.sqrt(torch.mean(feature[i][:length] **2, dim=0) - mean_vec[i] **2 + 1e-6)
            variance_vec_list.append(variances)
        var_vec = torch.stack(variance_vec_list)

        statistic_pooling = torch.cat([mean_vec, var_vec], dim=-1)

        return statistic_pooling