import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

class MaskedGumbelSigmoidLayer(nn.Module):
    """A layer that just applies a GumbelSigmoid nonlinearity.
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
    
    Args:
        mask (binary tensor): is a masked tensor, where 0 are the variants not selected
    Returns:
        _type_: a masked gumbel sigmoid layer, where each element is a continuous approximation of a bernoulli sample 
    """    
    """
    
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic (e.g. shared)
    eps: a small number used for numerical stability

    """
    def __init__(self, t=0.01, eps=1e-12, **kwargs):
        super(MaskedGumbelSigmoidLayer, self).__init__(**kwargs)
        self.sig = nn.Sigmoid()
        self.temperature = torch.tensor(t)
        self.eps = eps
        


    def forward(self, data):
        #sample from Gumbel(0, 1)
        input, mask = data # consider inputs are log(inputs)
        gumbels_part1 = torch.rand(input.shape)
        gumbels_part2 = torch.rand(input.shape)
        noise = -torch.log(torch.log(gumbels_part2 + self.eps)/torch.log(gumbels_part1 + self.eps) + self.eps)
        logits = self.sig((input+noise)/self.temperature)
        logits = torch.mul(logits,mask) + self.eps # enforce zero where not possible
        return torch.log(logits) # returns approximate sampels of a bernoulli(logits), so a binary where variants exist or do not exist


class BurdenSelector(nn.Module):
    def __init__(self, input_sz: int, hidden_szs: int, latent_sz: int, dropout_prob: float, num_layers: int):
        super(BurdenSelector, self).__init__()
        self.input_sz = input_sz # Max number of snps 
        self.hidden_szs = hidden_szs 
        self.latent_sz = latent_sz
        self.n_layer = num_layers
        self.selector_model = nn.Sequential()
        self.dropout_prob = dropout_prob
        self.sigmoid_layer = MaskedGumbelSigmoidLayer()
        self.selector_model.append(nn.Linear(self.input_sz, self.hidden_szs))
        self.sample_sigmoid_layer = nn.Sigmoid()
        self.eps = 1e-12
        for _ in range(self.n_layer - 2):
            self.selector_model.append(nn.Linear(self.hidden_szs, self.hidden_szs))
            self.selector_model.append(nn.SiLU())
            self.selector_model.append(nn.Dropout(self.dropout_prob))
        self.selector_model.append(nn.SiLU())
        self.selector_model.append(nn.Linear(self.hidden_szs,self.latent_sz))
        

    def forward(self, data):
        input, mask = data
        selectors_base = self.selector_model(input)
        #selectors = self.sample_sigmoid_layer(selectors_base)*mask+self.eps

        #selectors2 = self.sigmoid_layer([selectors_base, mask]) # using gumbel-sigmoid sampling 
        return selectors_base


    def bernoulli_sampling(self, logits):

        samples = torch.distributions.bernoulli.Bernoulli(probs=logits).sample()
        
        return samples