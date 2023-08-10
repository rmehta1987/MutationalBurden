import torch
import matplotlib.pyplot as plt
from burden_causal_effect import VariationalBurden
import os
from burden_csvdataset import CustomCSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bootstrap
import numpy as np

dataset_path = '30690NA_gene_effects_2'
model = torch.load('model_epoch_2')

num_zero_pad = 388 # for trait 30690
batch_size = 30
num_samples = 15
more_samples = 70

files = [os.path.join(dataset_path,file) for file in os.listdir(dataset_path)]
dataset = CustomCSVDataset(filenames = files, batch_size = 25, max_zero_pad=num_zero_pad)
dataloader = DataLoader(dataset,batch_size = None, shuffle = True)

shet = pd.read_csv('Dataset/s_het_estimates.genebayes_w_gene_names_3.tsv',header=0,delimiter='\t')

with torch.no_grad():
    print("Creating inference results dataframe")
    for i, data in enumerate(dataloader):

        beta_batch = data['beta']
        mask_batch = data['mask']
        locus = data['locus']
        standard_error_batch = data['se']
        gene = data['gene']
        ac = data['ac']
        af = data['af']
        beta_batch_shape, beta_col_shape = beta_batch.shape

        mask = mask_batch.unsqueeze(1).repeat(1, num_samples, 1)
        standard_error = standard_error_batch.unsqueeze(1).repeat(1, num_samples, 1)
        beta = beta_batch

        for k in range(0,more_samples): # this stacks more estiamted true betas on a specifc gene/sample, final shape should be [batch_shape, num_samples*more_samples, num_variants]

            posterior_beta, posterior_dist, critic_mean, critic_var, selector_logits, _ = model([beta, mask])
            spike = torch.mul(selector_logits.exp(), mask) + 1e-6
            spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
            q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=model.temperature, probs=spike)
            pi_samples = q_sampler.rsample()*mask # ignore non present variants
            posterior_beta = torch.mul(pi_samples,posterior_beta)
            if k > 0:
                more_posteriors = torch.cat((more_posteriors, posterior_beta),dim=1) # stack along sample dimension not batch dimension!
            else:
                more_posteriors = posterior_beta
            
        save_dict = dict()
        save_dict = {'genes': gene, 'post_betas': more_posteriors, 'af': af, 'mask_batch': mask_batch}


        torch.save(save_dict, f'30690_batch_{i}')

    '''
        for j in range(0,beta_batch.shape[0]):
            post_beta = posterior_beta[j].mean(axis=1) # generate bootstrap samples
            res_mean = bootstrap(post_beta.cpu().numpy(),np.mean, confidence_level=0.95)
            res_std = bootstrap(post_beta.cpu().numpy(),np.std, confidence_level=0.95)
            # Calculate burden mu = s_het*true_beta*sum_{k}_{kth_variant}[(af)]

            #check if gene exists
            check_gene = shet[shet['Gene']==gene[j]]
            if not check_gene.empty:
                burden = check_gene['post_mean'].iloc[0]*post_beta.cpu().numpy()*(torch.sum(af[j]).cpu().numpy())
                shet.at[check_gene.index[0],'U']=burden
                shet.at[check_gene.index[0],'post_beta']=post_beta.cpu().numpy()
                shet.at[check_gene.index[0],'sum_AF'] = torch.sum(af[j]).cpu().numpy()
        
    shet.to_csv('30690NA_shet_with_burden_and_true_beta_2.tsv',sep='\t', header=True, index=False)
    '''
