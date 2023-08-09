"""
This is a simple spike and slab latent regression using a variant of the
approach described in [1]. This approach is particularly suitable for situations
when assuming the true effect of a variants in a gene such as a loss of function 
is the same for all variants.

Estimated_beta ~ N(B_j, standard_error^2)
B_j ~ pi*N(0,sigma) + (1-pi)\diract(0) ----- this is the general spike and slab model

References
[1] False Discovery Rates: A New Deal
    Matthew Stephens
    bioRxiv 038216; doi: https://doi.org/10.1101/038216
    Now published in Biostatistics doi: 10.1093/biostatistics/kxw041
"""

import numpy as np
import torch
from torch.optim import Adam

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, SVI, MCMC, NUTS
from pyro.infer.autoguide import AutoDelta, init_to_median
from burden_csvdataset import CustomCSVDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from pyro import poutine

pyro.set_rng_seed(0)


def init_loc_fn(site):
    if site["beta"] == "weights":
        # Initialize weights to uniform.
        return torch.ones(1.)
    if site["rho"] == "scale":
        return torch.tensor(0.5)
    if site["pi"] == "locs":
        return torch.ones(1.)
    raise ValueError(site["name"])


def dot(X, Z):
    return torch.mm(X, Z.t())

# Most of the model code is concerned with constructing the sparsity inducing prior.
def model(Y, hypers):
    """_summary_

    Args:
        Y (_type_): _description_
        hypers (_type_): 
            - num_variants = number of variants/columns in observation
            - se = standard error of estimated effect sizes
            - 
    """    

    with pyro.plate("SpikeSlab", dim=-1):
        rho = pyro.sample("rho", dist.Beta(torch.ones(1,device=hypers["se"].device)*0.5, torch.ones(1,device=hypers["se"].device)*0.5))
        pi_spike_slab = pyro.sample("pi", dist.RelaxedBernoulliStraightThrough(temperature=torch.tensor(0.01,device=hypers["se"].device),probs=rho))
        #base_dist = dist.Normal(torch.zeros(1,device=hypers["se"].device), torch.ones(1,device=hypers["se"].device))
        #transform = dist.transforms.Spline(1, count_bins=8, bound=3.).to(hypers["se"].device)
        #pyro.module("beta_transform", transform)  
        #flow_dist = dist.TransformedDistribution(base_dist, [transform, transform])
        true_beta = pyro.sample("beta",dist.Normal(torch.zeros(1,device=hypers["se"].device),torch.ones(1,device=hypers["se"].device)))
        #true_beta = pyro.sample("beta",flow_dist)
    beta_loc = pi_spike_slab*true_beta
    mu = beta_loc*torch.ones(hypers["num_variants"],device=hypers["se"].device)
    with pyro.plate("slab",dim=-1):
        
        # observe the outputs Y
        pyro.sample(
            "obs",
            dist.Normal(mu, scale=hypers["se"].pow(2)),
            obs=Y,
        )
optim = pyro.optim.Adam({"lr": 0.1, "betas": [0.8, 0.99]})
elbo = TraceEnum_ELBO()
device=False


nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, warmup_steps=200, num_samples=500)
files = [os.path.join('30690NA_gene_effects_2',file) for file in os.listdir('30690NA_gene_effects_2')]
dataset = CustomCSVDataset(filenames = files, batch_size = 1, max_zero_pad=0)
dataloader = DataLoader(dataset,batch_size = None, shuffle = True)
approximated_true_betas=[]
for i, data in enumerate(dataloader):

    beta_batch = data['beta']
    mask_batch = data['mask']
    locus = data['locus']
    standard_error_batch = data['se']
    gene = data['gene']
    ac = data['ac']
    af = data['af']
    if device:
        hypers={"num_variants": beta_batch.shape[1], "se": standard_error_batch.cuda()}
        mcmc.run(beta_batch.cuda(),hypers)
    else:
        hypers={"num_variants": beta_batch.shape[1], "se": standard_error_batch}
        mcmc.run(beta_batch,hypers)
    mcmc.summary(prob=0.8)
    posterior_samples = mcmc.get_samples()
    beta_mu = posterior_samples['beta'].cpu().numpy()
    pi_s = posterior_samples['pi'].cpu().numpy()

    info = {'mu': beta_mu, 'pi': pi_s, 'gene': gene}
    approximated_true_betas.append(info)

    if i % 10 == 0:

        posterior_samples = mcmc.get_samples()
        beta_mu = torch.mean(posterior_samples['beta'], axis=0)
        mu = beta_mu*torch.ones(hypers["num_variants"],device=hypers["se"].device)
        recon_beta = dist.Normal(mu, hypers["se"].pow(2))
        recon_beta_samps = recon_beta.sample().cpu().numpy()



        #plt.step(range(len(beta_mu)), beta_mu,  where='mid', lw=1)
        plt.plot(range(len(recon_beta_samps[0])), recon_beta_samps[0], 'r')
        plt.plot(range(len(recon_beta_samps[0])), beta_batch[0], 'b')
        plt.ylabel(r'$\beta$')
        plt.xlabel('locus')
        plt.savefig('pyro_mcmc_true_beta_{}.png'.format(i))
        plt.close()
        results = np.asarray(approximated_true_betas)
        np.save('results_mcmc_{}'.format(i), results)


    

results = np.asarray(approximated_true_betas)
np.save('results_mcmc_final', results)