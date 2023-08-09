import torch
from burden_causal_effect import VariationalBurden
import os
from burden_csvdataset import CustomCSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchview import draw_graph
import graphviz
from torch.masked import masked_tensor, as_masked_tensor
import pandas as pd

graphviz.set_jupyter_format('png')

def run_model(dataset_path: str, device='cpu') -> None:
    """_summary_

    Args:
        dataset_path (str): _description_
    """    
    shet = pd.read_csv('Dataset/s_het_estimates.genebayes_w_gene_names_2.tsv',header=0,delimiter='\t')
    shet['U']=0
    shet['post_beta']=0
    shet['sum_AF']=0
    num_zero_pad = 388 # for trait 30690
    batch_size = 30
    num_samples = 30
    #num_sero_pad = 328 # for bioplar
    files = [os.path.join(dataset_path,file) for file in os.listdir(dataset_path)]
    dataset = CustomCSVDataset(filenames = files, batch_size = batch_size, max_zero_pad=num_zero_pad)
    dataloader = DataLoader(dataset,batch_size = None, shuffle = True)

    # Set up model
    model = VariationalBurden(num_variants=num_zero_pad, hidden_sz=2*num_zero_pad, latent_sz=num_zero_pad, num_samples=num_samples,  batch_size=batch_size, lambda_param=0.1, num_transforms=2, num_blocks=2, 
                              dropout_probability=0.0, permute_flow=False, use_batch_norm=False,num_bins=16,tail_bound=3, device=False)
    dict_of_params = model.model_parameters()

    prior_opt = torch.optim.Adam(dict_of_params['prior'], lr=1e-3)
    encoder_opt = torch.optim.Adam(dict_of_params['encoder'], lr=1e-3, betas=(0.90, 0.999))
    epoch = 0
    max_num_epochs = 1
    running_critic_loss = 0
    running_selector_loss = 0
    running_latent_loss = 0
    running_slab_loss = 0
    running_spike_loss = 0
    running_recon_loss = 0
    total_iter=0
    
    kl_slab_losses = []
    kl_spike_losses = []
    recon_losses = []
    latent_losses = []
    while epoch <= max_num_epochs:
        model.train()
        for i, data in enumerate(dataloader):

            beta_batch = data['beta']
            mask_batch = data['mask']
            locus = data['locus']
            standard_error_batch = data['se'].to(device)
            gene = data['gene']
            ac = data['ac']
            af = data['af']
            beta_batch_shape, beta_col_shape = beta_batch.shape

            prior_opt.zero_grad()
            encoder_opt.zero_grad()

            #beta = beta_batch.expand(num_samples,beta_batch_shape,-1).clone()
            mask = mask_batch.unsqueeze(1).repeat(1, num_samples, 1)
            standard_error = standard_error_batch.unsqueeze(1).repeat(1, num_samples, 1)

            #beta = beta.reshape(num_samples*beta_batch_shape,beta_col_shape)
            #mask = mask.reshape(num_samples*beta_batch_shape,beta_col_shape)
            #standard_error = standard_error.reshape(num_samples*beta_batch_shape,beta_col_shape)

            beta = beta_batch
            #mask = mask_batch
            #standard_error = standard_error_batch

            #if i == 0:
            #    model_graph = draw_graph(model, input_data=[beta, mask], 
            #                                                graph_name='burden2', depth=3, save_graph=True, 
            #                                                expand_nested=True, roll=True, device='cpu', hide_inner_tensors=True, hide_module_functions=False, standard_error=standard_error, num_samples=num_samples, beta_batch_shape=beta_batch_shape)
            #    model.train()
                                                                          

            posterior_beta, posterior_dist, critic_mean, critic_var, selector_logits, selected = model([beta, mask])            
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var, num_samples, beta_batch_shape) 

            #latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_flow_and_decoder_no_mask(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var, num_samples, beta_batch_shape)

            latent_loss, beta_hat_loss, kl_slab, kl_spike, \
                 posterior_beta, true_beta, mean_estimate, \
                     spikes, prior_dist = model.iwae_latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, 
                                                                                                  posterior_beta, posterior_dist, selector_logits, 
                                                                                                  selected, critic_mean, critic_var, num_samples, beta_batch_shape)

            #latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_spike_and_decoder_no_mask(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var, num_samples, beta_batch_shape)

            #latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_spike_and_decoder_masked_tensor(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var, num_samples, beta_batch_shape)


            running_latent_loss += latent_loss.detach()
            running_spike_loss += torch.mean(kl_spike.detach().sum(axis=0))
            running_slab_loss += torch.mean(kl_slab.detach().sum(axis=0))
            running_recon_loss += torch.mean(beta_hat_loss.detach().sum(axis=0))
            total_iter+=1

            latent_loss.backward()

            torch.nn.utils.clip_grad_norm_(dict_of_params['prior'], 1., norm_type=2)
            
            #torch.nn.utils.clip_grad_norm_(dict_of_params['selector'], 1., norm_type=2)
            prior_opt.step()
            encoder_opt.step()
            model.latent_prior.a_dist.clear_cache()
            #selector_opt.step()
            
            recon_losses.append(torch.mean(beta_hat_loss.detach().sum(axis=0)))
            kl_spike_losses.append(torch.mean(kl_spike.detach().sum(axis=0)))
            kl_slab_losses.append(torch.mean(kl_slab.detach().sum(axis=0)))
            latent_losses.append(latent_loss.detach())
            temp = model.latent_prior.sample()
            if torch.any(torch.isnan(temp)):
                print("WTF is samples from latent prior are invalid")


            if (i+1) % 10== 0:
                print("On iteration: {} with total loss {}, with loss of reconstruction of estimated betas: {} spike KL loss: {}, slab KL (selection of variants) loss: {}".format(
                    i+1, running_latent_loss/(i+1), running_recon_loss/(i+1), running_spike_loss/(i+1), running_slab_loss/(i+1)))
                idx = np.random.randint(0,beta_batch_shape)
                predicted_spikes = spikes[idx].sum(axis=1).mean(axis=0)

                print("Number of expected variants is {}, number predicted on average is {}".format(mask_batch[idx].sum(),predicted_spikes))
                #print("Average value of true_beta predicted: {}".format(posterior_beta.mean(axis=0)[:25]))
                #print("Average value of recon_beta predicted: {}".format(mean_estimate.mean(axis=0)[:25]))
                #prior_samps = prior_dist.sample((1000,)).mean(axis=0)
                #print("Average value of prior: {}".format(prior_samps[:20].detach()))
                #print("Number of expected variants is 9, number predicted on average is {}".format(torch.round(spikes.detach())[0,:10]))

                print('\n')
            if (i+1) % 50 == 0:

                # Beta inference plots
                
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(2, 1) # 2 rows, 2 columns
                idx = np.random.randint(0,beta_batch_shape)
                
                predicted_spikes = spikes[idx].sum(axis=1).mean(axis=0)
                print("Number of expected variants is {}, number predicted on average is {}".format(mask_batch[idx].sum(),predicted_spikes))
                
                post_data = posterior_beta[idx].mean(axis=0)
                ax1 = fig.add_subplot(gs[0,0])
                
                if model.latent_sz==1:
                    ax1.axhline(y=posterior_beta[idx:-1:beta_batch_shape].mean(axis=0).numpy(), c='r', label='Posterior Beta')
                else:
                    ax1.plot(post_data[:25].numpy(),'--', c='r', label='Posterior Beta')
                   
                ax1.plot(beta_batch[idx, :25],c='b', label='Observed Beta on Sample {}'.format(idx))
                #ax1.legend(loc='lower right')
                ax1.set_ylabel('Posterior Beta on sample {}'.format(idx))
                ax1.set_xlabel('Locus')
                mean_data = mean_estimate[idx].mean(axis=0)
                
                ax3 = fig.add_subplot(gs[1,0])                
                ax3.plot(mean_data[:25].numpy(),'--', c='r', label='Reconstructed Beta')
                ax3.plot(beta_batch[idx, :25],c='b', label='Observed Beta')
                #ax3.legend(loc='lower right')
                ax3.set_ylabel('Reconstructed Beta on sample {}'.format(idx))
                ax3.set_xlabel('Locus')

                #mean_prediction = mean_estimate[idx:-1:batch_size].mean(axis=0).numpy()
                #percentiles = np.percentile(mean_estimate[idx:-1:batch_size].numpy(), [5.0, 95.0], axis=0)
                #ax4 = fig.add_subplot(gs[2,0])
                #ax4.fill_between(beta_batch[idx, :25], percentiles[0, :25], percentiles[1, :25], color="lightblue")
                # plot mean prediction
                #ax4.plot(beta_batch[idx, :25], mean_prediction[:25], "blue", ls="solid", lw=2.0)
           
                
                plt.savefig('simulated_beta_inferences_epoch_{}_iter{}.jpg'.format(epoch,i))                
                plt.close()
                # Spike inference plots
                '''
                selector_logits = torch.log(selector_logits.detach())
                plt.title('Spike and Slabs')
                plt.plot(selector_logits[0,:60], c='r', label='Spikes')
                plt.savefig('spikes_savefig_epoch_{}_iter{}.jpg'.format(epoch,i))
                plt.legend(loc='upper left')
                plt.close()
                '''
                # with torch.no_grad():
                #     spike = torch.mul(selector_logits.exp(), mask) + 1e-6
                #     spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
                #     q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=model.temperature, probs=spike)
                #     pi_samples = q_sampler.rsample()*mask # ignore non present variants
                #     posterior_beta = torch.mul(pi_samples,posterior_beta)
                #     for j in range(0,beta_batch.shape[0]):
                #         post_beta = posterior_beta[j].sum(axis=1)/mask[j,0].sum() # get mean of beta's across variants
                #         post_beta = post_beta.mean(axis=0) # get mean of beta across samples
                #         # Calculate burden mu = s_het*true_beta*sum_{k}_{kth_variant}[(af)]

                #         #check if gene exists
                #         check_gene = shet[shet['Gene']==gene[j]]
                #         if not check_gene.empty:
                #             burden = check_gene['post_mean'].iloc[0]*post_beta.cpu().numpy()*(torch.sum(af[j]).cpu().numpy())
                #             shet.at[check_gene.index[0],'U']=burden
                #             shet.at[check_gene.index[0],'post_beta']=post_beta.cpu().numpy()
                #             shet.at[check_gene.index[0],'sum_AF'] = torch.sum(af[j]).cpu().numpy()
                    
                # shet.to_csv('shet_with_burden_and_true_beta.tsv',sep='\t', header=True, index=False)


            

        print("Finished Epoch: {} with loss of reconstruction of estimated betas: {} spike KL loss: {}, slab KL (selection of variants) loss: {}".format(
                    epoch, running_latent_loss/total_iter, running_spike_loss/total_iter, running_slab_loss/total_iter))
        
        
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Losses Recon (r), KL_spike (b), KL_slab (g)')
        gs = fig.add_gridspec(2, 2) # 2 rows, 2 columns
        ax1 = fig.add_subplot(gs[1,:])        
        ax1.plot(np.asarray(recon_losses), c='r', label="Recon")

        ax2 = fig.add_subplot(gs[0,0])        
        ax2.plot(np.asarray(kl_spike_losses),c='g', label='Slab')

        ax3 = fig.add_subplot(gs[0,1])       
        ax3.plot(np.asarray(kl_spike_losses),c='b', label='Spike')
        plt.savefig('simulated_losses_epoch_{}_iter{}.jpg'.format(epoch,i))
        #plt.legend(loc='upper left')
        epoch += 1
        plt.close()
        ## Create estimates of "mutational burden"
        

        if epoch > max_num_epochs:
            torch.save(model, f'model_epoch_{epoch}')

        running_critic_loss = 0
        running_selector_loss = 0
        running_latent_loss = 0
        total_iter=0
        kl_slab_losses = []
        kl_spike_losses = []
        recon_losses = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            beta_batch = data['beta']
            mask_batch = data['mask']
            locus = data['locus']
            standard_error_batch = data['se'].to(device)
            gene = data['gene']
            ac = data['ac']
            af = data['af']
            beta_batch_shape, beta_col_shape = beta_batch.shape

            mask = mask_batch.unsqueeze(1).repeat(1, num_samples, 1)
            standard_error = standard_error_batch.unsqueeze(1).repeat(1, num_samples, 1)
            beta = beta_batch


            posterior_beta, posterior_dist, critic_mean, critic_var, selector_logits, _ = model([beta, mask])
            spike = torch.mul(selector_logits.exp(), mask) + 1e-6
            spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
            q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=model.temperature, probs=spike)
            pi_samples = q_sampler.rsample()*mask # ignore non present variants

            posterior_beta = torch.mul(pi_samples,posterior_beta)
            for j in range(0,beta_batch.shape[0]):
                post_beta = posterior_beta[j].mean(axis=1).mean(axis=0)
                # Calculate burden mu = s_het*true_beta*sum_{k}_{kth_variant}[(af)]

                #check if gene exists
                check_gene = shet[shet['Gene']==gene[j]]
                if not check_gene.empty:
                    burden = check_gene['post_mean'].iloc[0]*post_beta.cpu().numpy()*(torch.sum(af[j]).cpu().numpy())
                    shet.at[check_gene.index[0],'U']=burden
                    shet.at[check_gene.index[0],'post_beta']=post_beta.cpu().numpy()
                    shet.at[check_gene.index[0],'sum_AF'] = torch.sum(af[j]).cpu().numpy()
            
        shet.to_csv('30690NA_shet_with_burden_and_true_beta.tsv',sep='\t', header=True, index=False)

                

    
    


    


def create_simualated_data():

    true_weights = torch.tensor(np.array([-4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.2, 0, 37.1, 0, 0, 50, - 0.00005, 10, 3, 0])).float()
    true_se = torch.tensor(np.array([-1.53656726, -1.12270527,  1.33041824, -0.8407924 ,  0.12089116,
       -1.02052477,  0.63523762, -1.2773014 , -0.14838007,  0.55948691,
        0.0581737 ,  1.14084851, -1.3167906 , -0.80398104,  1.61177848,
       -0.89317884, -0.54195322,  0.99309246, -1.15119029, -0.39033024])).pow(2)
    estimated_weights_sampler = torch.distributions.multivariate_normal.MultivariateNormal(true_weights, torch.eye(true_weights.shape[0]))
    estimated_weights = estimated_weights_sampler.sample((10000,))

    return true_weights, estimated_weights, true_se

def run_model_test():
    """_summary_

    Args:
        dataset_path (str): _description_
    """    

    num_zero_pad = 358 
    true, estimated, true_se = create_simualated_data()
    true_zero_padded = torch.nn.functional.pad(true,pad=(0,num_zero_pad-len(true)),value=0).float()
    estimated_zero_padded = torch.nn.functional.pad(estimated,pad=(0,num_zero_pad-estimated.shape[1]),value=0).float()
    mask = (~(estimated_zero_padded==0)).float()
    #true_se_padded = true_se.unsqueeze(1)*mask
    estimated_dataset = torch.tensor_split(estimated_zero_padded, 500)
    mask_dataset = torch.tensor_split(mask, 500)
    #true_se_dataset = torch.tensor_split(true_se_padded, 500)

    # Set up model
    model = VariationalBurden(num_variants=num_zero_pad, hidden_sz=2*num_zero_pad, latent_sz=num_zero_pad, batch_size=50, lambda_param=0.1, num_transforms=2, num_blocks=2, 
                              dropout_probability=0.0, permute_flow=True, use_batch_norm=False,num_bins=8,tail_bound=4, device=False)
    dict_of_params = model.model_parameters()
    #selector_opt = torch.optim.Adam(dict_of_params['selector'], lr=1e-3, weight_decay=1e-2)
    prior_opt = torch.optim.Adam(dict_of_params['prior'], lr=1e-3, weight_decay=1e-2)
    encoder_opt = torch.optim.Adam(dict_of_params['encoder'], lr=1e-3, betas=(0.90, 0.999), weight_decay=1e-2)
    epoch = 0
    max_num_epochs = 20
    running_critic_loss = 0
    running_selector_loss = 0
    running_latent_loss = 0
    running_slab_loss = 0
    running_spike_loss = 0
    running_recon_loss = 0
    total_iter=0
    kl_slab_losses = []
    kl_spike_losses = []
    recon_losses = []
    while epoch <= max_num_epochs:
        model.train()
        for i, (beta, mask) in enumerate(zip(estimated_dataset, mask_dataset)):

            prior_opt.zero_grad()
            encoder_opt.zero_grad()

            beta = beta.expand(10,beta.shape[0],-1)
            mask = mask.expand(10,mask.shape[0],-1)
            standard_error = mask.clone()

            posterior_beta, posterior_dist, critic_mean, critic_var, selector_logits, selected = model([beta, mask])            
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate = model.latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate, spikes, prior_dist = model.latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            
            # latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var)
            
            latent_loss, beta_hat_loss, kl_slab, kl_spike, \
                 posterior_beta, true_beta, mean_estimate, \
                     spikes, prior_dist = model.iwae_latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, 
                                                                                                  posterior_beta, posterior_dist, selector_logits, 
                                                                                                  selected, critic_mean, critic_var) 
                                      

            running_latent_loss += latent_loss.detach()
            running_spike_loss += torch.mean(kl_spike.detach())
            running_slab_loss += torch.mean(kl_slab.detach())
            running_recon_loss += torch.mean(beta_hat_loss.detach())

            latent_loss.backward()

            #torch.nn.utils.clip_grad_norm_(dict_of_params['prior'], 1., norm_type=2)
            
            #torch.nn.utils.clip_grad_norm_(dict_of_params['selector'], 1., norm_type=2)
            #prior_opt.step()
            encoder_opt.step()
            #selector_opt.step()
            
            recon_losses.append(torch.mean(beta_hat_loss,axis=0).numpy())
            kl_spike_losses.append(torch.mean(kl_spike,axis=0).numpy())
            kl_slab_losses.append(torch.mean(kl_slab,axis=0).numpy())
            temp = model.latent_prior.rsample()
            if torch.any(torch.isnan(temp)):
                print("WTF is samples from latent prior are invalid")


            if (i+1) % 10== 0:
                print("On iteration: {} with loss of reconstruction of estimated betas: {} spike KL loss: {}, slab KL (selection of variants) loss: {}".format(
                    i+1, running_recon_loss/i, running_spike_loss/i, running_slab_loss/i))
                print("Number of expected variants is 9, number predicted on average is {}".format(torch.sum(torch.round(spikes.detach()),axis=1).mean(axis=0)))
                print("Average value of true_beta predicted: {}".format(posterior_beta.mean(axis=0)[:10]))
                print("Average value of recon_beta predicted: {}".format(mean_estimate.mean(axis=0)[:10]))
                prior_samps = prior_dist.rsample()
                print("Average value of prior: {}".format(prior_samps[:10].detach()))
                print("Number of expected variants is 9, number predicted on average is {}".format(torch.round(spikes.detach())[0,:10]))

                print('\n')
            if (i+1) % 2 == 0:

                # Beta inference plots
                
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(2, 1) # 2 rows, 2 columns

                # Posterior Beta
                ax1 = fig.add_subplot(gs[0,0])
                ax1.plot(posterior_beta.mean(axis=0).mean(axis=0)[:25],'--', c='r', label='Posterior Beta')
                #ax1.plot(beta[0, :25],c='b', label='Observed Beta')
                ax1.plot(true_zero_padded[:25],c='g', label='True Beta')
                #ax1.legend(loc='lower right')
                ax1.set_ylabel('Beta')
                ax1.set_xlabel('Locus')
                
                ax3 = fig.add_subplot(gs[1,0])
                ax3.plot(mean_estimate.mean(axis=0).mean(axis=0)[:25],'--', c='r', label='Reconstructed Beta')
                ax3.plot(beta.mean(axis=0).mean(axis=0)[:25],c='b', label='Observed Beta')
                #ax3.legend(loc='lower right')
                ax3.set_ylabel('Beta')
                ax3.set_xlabel('Locus')
           
                
                plt.savefig('simulated_beta_inferences_epoch_{}_iter{}.jpg'.format(epoch,i))                
                plt.close()
                # Spike inference plots
                '''
                selector_logits = torch.log(selector_logits.detach())
                plt.title('Spike and Slabs')
                plt.plot(selector_logits[0,:60], c='r', label='Spikes')
                plt.savefig('spikes_savefig_epoch_{}_iter{}.jpg'.format(epoch,i))
                plt.legend(loc='upper left')
                plt.close()
                '''

            total_iter+=1

        print("Epoch: {} with loss of reconstruction of estimated betas: {} critic loss: {}, selector (selection of variants) loss: {}".format(
            epoch, running_latent_loss/i, running_critic_loss/i, running_selector_loss/i))
        
        epoch += 1
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Losses Recon (r), KL_spike (b), KL_slab (g)')
        gs = fig.add_gridspec(2, 2) # 2 rows, 2 columns
        ax1 = fig.add_subplot(gs[1,:])        
        ax1.plot(np.asarray(recon_losses), c='r', label="Recon")

        ax2 = fig.add_subplot(gs[0,0])        
        ax2.plot(np.asarray(kl_spike_losses),c='g', label='Slab')

        ax3 = fig.add_subplot(gs[0,1])       
        ax3.plot(np.asarray(kl_spike_losses),c='b', label='Spike')
        plt.savefig('simulated_losses_epoch_{}_iter{}.jpg'.format(epoch,i))
        #plt.legend(loc='upper left')
        
        plt.close()
        if epoch > max_num_epochs:
            torch.save(model, f'model_epoch_{epoch}')

        running_critic_loss = 0
        running_selector_loss = 0
        running_latent_loss = 0
        total_iter=0
        kl_slab_losses = []
        kl_spike_losses = []
        recon_losses = []

def run_model_test_iwae():
    """_summary_

    Args:
        dataset_path (str): _description_
    """    

    num_zero_pad = 358 
    true, estimated, true_se = create_simualated_data()
    true_zero_padded = torch.nn.functional.pad(true,pad=(0,num_zero_pad-len(true)),value=0).float()
    estimated_zero_padded = torch.nn.functional.pad(estimated,pad=(0,num_zero_pad-estimated.shape[1]),value=0).float()
    mask = (~(estimated_zero_padded==0)).float()
    #true_se_padded = true_se.unsqueeze(1)*mask
    estimated_dataset = torch.tensor_split(estimated_zero_padded, 500)
    mask_dataset = torch.tensor_split(mask, 500)
    #true_se_dataset = torch.tensor_split(true_se_padded, 500)

    # Set up model
    model = VariationalBurden(num_variants=num_zero_pad, hidden_sz=2*num_zero_pad, latent_sz=num_zero_pad, batch_size=50, lambda_param=0.1, num_transforms=2, num_blocks=2, 
                              dropout_probability=0.0, permute_flow=False, use_batch_norm=False,num_bins=8,tail_bound=4, device=False)
    dict_of_params = model.model_parameters()
    #selector_opt = torch.optim.Adam(dict_of_params['selector'], lr=1e-3, weight_decay=1e-2)
    prior_opt = torch.optim.Adam(dict_of_params['prior'], lr=1e-3, weight_decay=1e-2)
    encoder_opt = torch.optim.Adam(dict_of_params['encoder'], lr=1e-3, betas=(0.90, 0.999), weight_decay=1e-2)
    epoch = 0
    max_num_epochs = 20
    running_critic_loss = 0
    running_selector_loss = 0
    running_latent_loss = 0
    running_slab_loss = 0
    running_spike_loss = 0
    running_recon_loss = 0
    total_iter=0
    kl_slab_losses = []
    kl_spike_losses = []
    recon_losses = []
    num_samples = 5
    model.train()
    while epoch <= max_num_epochs:
        
        for i, (beta, mask) in enumerate(zip(estimated_dataset, mask_dataset)):

            beta_batch, beta_col = beta.shape
            mask_batch, mask_col = mask.shape
            true_beta_batch = beta


            prior_opt.zero_grad()
            encoder_opt.zero_grad()

            beta = beta.expand(num_samples,beta_batch,-1)
            mask = mask.expand(num_samples,mask_batch,-1)
            beta = beta.reshape(num_samples*beta_batch,beta_col)
            mask = mask.reshape(num_samples*mask_batch,mask_col)
            standard_error = mask.clone()

            posterior_beta, posterior_dist, critic_mean, critic_var, selector_logits, selected = model([beta, mask])            
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate = model.latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate, spikes, prior_dist = model.latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, \
            #     posterior_beta, true_beta, mean_estimate, \
            #         spikes, prior_dist = model.iwae_latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, 
            #                                                                                      posterior_beta, posterior_dist, selector_logits, 
            #                                                                                      selected, critic_mean, critic_var, num_samples, beta_batch)
            
            
            latent_loss, beta_hat_loss, kl_slab, kl_spike, \
                 posterior_beta, true_beta, mean_estimate, \
                     spikes, prior_dist = model.iwae_latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, 
                                                                                                  posterior_beta, posterior_dist, selector_logits, 
                                                                                                  selected, critic_mean, critic_var, num_samples, beta_batch) 
                                      

            running_latent_loss += latent_loss.detach()
            running_spike_loss += torch.mean(kl_spike.detach())
            running_slab_loss += torch.mean(kl_slab.detach())
            running_recon_loss += torch.mean(beta_hat_loss.detach())

            latent_loss.backward()

            torch.nn.utils.clip_grad_norm_(dict_of_params['prior'], 1., norm_type=2)
            
            #torch.nn.utils.clip_grad_norm_(dict_of_params['selector'], 1., norm_type=2)
            prior_opt.step()
            encoder_opt.step()
            #selector_opt.step()
            
            recon_losses.append(torch.mean(beta_hat_loss,axis=0).numpy())
            kl_spike_losses.append(torch.mean(kl_spike,axis=0).numpy())
            kl_slab_losses.append(torch.mean(kl_slab,axis=0).numpy())
            temp = model.latent_prior.rsample()
            if torch.any(torch.isnan(temp)):
                print("WTF is samples from latent prior are invalid")


            if (i+1) % 10== 0:
                print("On iteration: {} with loss of reconstruction of estimated betas: {} spike KL loss: {}, slab KL (selection of variants) loss: {}".format(
                    i+1, running_recon_loss/i, running_spike_loss/i, running_slab_loss/i))
                exp_spikes = spikes.detach().sum(axis=0)*1/spikes.shape[0]


                print("Number of expected variants is 9, number predicted on average is {}".format(exp_spikes[:35]))
                #print("Average value of true_beta predicted: {}".format(posterior_beta.mean(axis=0)[:25]))
                #print("Average value of recon_beta predicted: {}".format(mean_estimate.mean(axis=0)[:25]))
                prior_samps = prior_dist.sample((1000,)).mean(axis=0)
                print("Average value of prior: {}".format(prior_samps[:20].detach()))
                #print("Number of expected variants is 9, number predicted on average is {}".format(torch.round(spikes.detach())[0,:10]))

                print('\n')
            if (i+1) % 50 == 0:

                # Beta inference plots
                
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(2, 1) # 2 rows, 2 columns

                # Posterior Beta
                ax1 = fig.add_subplot(gs[0,0])
                ax1.plot(posterior_beta.mean(axis=0)[:25],'--', c='r', label='Posterior Beta')
                #ax1.plot(beta[0, :25],c='b', label='Observed Beta')
                ax1.plot(true_zero_padded[:25],c='g', label='True Beta')
                #ax1.legend(loc='lower right')
                ax1.set_ylabel('Beta')
                ax1.set_xlabel('Locus')
                
                ax3 = fig.add_subplot(gs[1,0])
                ax3.plot(mean_estimate.mean(axis=0)[:25],'--', c='r', label='Reconstructed Beta')
                ax3.plot(true_beta_batch.mean(axis=0)[:25],c='b', label='Observed Beta')
                #ax3.legend(loc='lower right')
                ax3.set_ylabel('Beta')
                ax3.set_xlabel('Locus')
           
                
                plt.savefig('simulated_beta_inferences_epoch_{}_iter{}.jpg'.format(epoch,i))                
                plt.close()
                # Spike inference plots
                '''
                selector_logits = torch.log(selector_logits.detach())
                plt.title('Spike and Slabs')
                plt.plot(selector_logits[0,:60], c='r', label='Spikes')
                plt.savefig('spikes_savefig_epoch_{}_iter{}.jpg'.format(epoch,i))
                plt.legend(loc='upper left')
                plt.close()
                '''

            total_iter+=1

        print("Epoch: {} with loss of reconstruction of estimated betas: {} critic loss: {}, selector (selection of variants) loss: {}".format(
            epoch, running_latent_loss/i, running_critic_loss/i, running_selector_loss/i))
        
        epoch += 1
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Losses Recon (r), KL_spike (b), KL_slab (g)')
        gs = fig.add_gridspec(2, 2) # 2 rows, 2 columns
        ax1 = fig.add_subplot(gs[1,:])        
        ax1.plot(np.asarray(recon_losses), c='r', label="Recon")

        ax2 = fig.add_subplot(gs[0,0])        
        ax2.plot(np.asarray(kl_spike_losses),c='g', label='Slab')

        ax3 = fig.add_subplot(gs[0,1])       
        ax3.plot(np.asarray(kl_spike_losses),c='b', label='Spike')
        plt.savefig('simulated_losses_epoch_{}_iter{}.jpg'.format(epoch,i))
        #plt.legend(loc='upper left')
        
        plt.close()
        if epoch > max_num_epochs:
            torch.save(model, f'model_epoch_{epoch}')

        running_critic_loss = 0
        running_selector_loss = 0
        running_latent_loss = 0
        total_iter=0
        kl_slab_losses = []
        kl_spike_losses = []
        recon_losses = []

    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_model('30690NA_gene_effects_2')
    #run_model_test_iwae()
