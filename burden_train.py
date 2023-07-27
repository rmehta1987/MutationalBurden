import torch
from burden_causal_effect import VariationalBurden
import os
from csvdataset import CustomCSVDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def run_model(dataset_path: str, device='cpu') -> None:
    """_summary_

    Args:
        dataset_path (str): _description_
    """    

    num_zero_pad = 358 
    #num_sero_pad = 328 # for bioplar
    files = [os.path.join(dataset_path,file) for file in os.listdir(dataset_path)]
    dataset = CustomCSVDataset(filenames = files, batch_size = 50, max_zero_pad=num_zero_pad)
    dataloader = DataLoader(dataset,batch_size = None, shuffle = True)

    # Set up model
    model = VariationalBurden(num_variants=num_zero_pad, hidden_sz=2*num_zero_pad, latent_sz=num_zero_pad, batch_size=50, lambda_param=0.1, num_transforms=2, num_blocks=2, 
                              dropout_probability=0.0, permute_flow=True, use_batch_norm=False,num_bins=8,tail_bound=3, device=False)
    dict_of_params = model.model_parameters()

    selector_optimizer = torch.optim.Adam(dict_of_params['selector'], lr=1e-3, betas=(0.90, 0.999), weight_decay=1e-2)
    #only_critic_opt = torch.optim.Adam(set(dict_of_params['critic']), lr=1e-3, weight_decay=1e-2)
    only_latent_opt = torch.optim.Adam(dict_of_params['latent'], lr=1e-3, weight_decay=1e-2)
    epoch = 0
    max_num_epochs = 5
    running_critic_loss = 0
    running_selector_loss = 0
    running_latent_loss = 0
    running_slab_loss = 0
    running_spike_loss = 0
    running_recon_loss = 0
    total_iter=0
    K=20
    kl_slab_losses = []
    kl_spike_losses = []
    recon_losses = []
    while epoch <= max_num_epochs:
        model.train()
        for i, data in enumerate(dataloader):

            selector_optimizer.zero_grad()
            #critic_base_optimizer.zero_grad()
            only_latent_opt.zero_grad()

            beta = data['beta'].to(device)
            mask = data['mask'].to(device)
            locus = data['locus']
            standard_error = data['se'].to(device)
            gene = data['gene']

            posterior_beta, critic_mean, critic_var, selector_logits, selected = model([beta, mask])            
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate = model.latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate = model.latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, posterior_beta, 
                                                                                                                              selector_logits, selected, critic_mean, critic_var)
            
            
            running_latent_loss += latent_loss.detach()
            running_spike_loss += torch.mean(kl_spike.detach())
            running_slab_loss += torch.mean(kl_slab.detach())
            running_recon_loss += torch.mean(beta_hat_loss.detach())
            latent_loss.backward()
            selector_optimizer.step()
            only_latent_opt.step()
            recon_losses.append(torch.mean(beta_hat_loss,axis=0).numpy())
            kl_spike_losses.append(torch.mean(kl_spike,axis=0).numpy())
            kl_slab_losses.append(torch.mean(kl_slab,axis=0).numpy())


            if (i+1) % 20 == 0:
                print("On iteration: {} with loss of reconstruction of estimated betas: {} spike KL loss: {}, slab KL (selection of variants) loss: {}".format(
                    i+1, running_recon_loss/i, running_spike_loss/i, running_slab_loss/i))
                #print(beta_hat[0,:20])
                #print(beta[0, :20])
                print('\n')
            if (i+1) % 50 == 0:

                # Beta inference plots
                
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(2, 1) # 2 rows, 2 columns

                # Posterior Beta
                ax1 = fig.add_subplot(gs[0,0])
                #ax1.set_title('Posterior Beta (r) vs Observed Beta (b)')
                ax1.plot(posterior_beta[0,:60],'--', c='r', label='Posterior Beta')
                ax1.plot(beta[0, :60],c='b', label='Observed Beta')
                ax1.legend(loc='lower right')
                ax1.set_ylabel('Beta')
                ax1.set_xlabel('Locus')
                
                ax3 = fig.add_subplot(gs[1,0])
                #ax1.set_title('Posterior Beta (r) vs Observed Beta (b)')
                ax3.plot(mean_estimate[0,:60],'--', c='r', label='Reconstructed Beta')
                ax3.plot(beta[0, :60],c='b', label='Observed Beta')
                ax3.legend(loc='lower right')
                ax3.set_ylabel('Beta')
                ax3.set_xlabel('Locus')
           
                
                plt.savefig('beta_inferences_epoch_{}_iter{}.jpg'.format(epoch,i))                
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
        plt.savefig('losses_epoch_{}_iter{}.jpg'.format(epoch,i))
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

def create_simualated_data():

    true_weights = torch.tensor(np.array([-4, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.2, 0, 37.1, 0, 0, 50, - 0.00005, 10, 3, 0]))
    true_se = torch.tensor(np.array([-1.53656726, -1.12270527,  1.33041824, -0.8407924 ,  0.12089116,
       -1.02052477,  0.63523762, -1.2773014 , -0.14838007,  0.55948691,
        0.0581737 ,  1.14084851, -1.3167906 , -0.80398104,  1.61177848,
       -0.89317884, -0.54195322,  0.99309246, -1.15119029, -0.39033024])).pow(2)
    estimated_weights_sampler = torch.distributions.normal.Normal(true_weights, true_se)
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

    #selector_optimizer = torch.optim.Adam(dict_of_params['selector'], lr=1e-3, betas=(0.90, 0.999), weight_decay=1e-2)
    only_latent_opt = torch.optim.Adam(dict_of_params['latent'], lr=1e-3, weight_decay=1e-2)
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

            #selector_optimizer.zero_grad()
            only_latent_opt.zero_grad()

            standard_error = mask.clone()

            posterior_beta, critic_mean, critic_var, selector_logits, selected = model([beta, mask])            
            
            #latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate = model.latent_variable_update_with_spike_and_decoder(beta, mask, standard_error, posterior_beta, 
            #                                                                                                                  selector_logits, selected, critic_mean, critic_var)
            latent_loss, beta_hat_loss, kl_slab, kl_spike, posterior_beta, true_beta, mean_estimate, spikes = model.latent_variable_update_with_flow_and_decoder(beta, mask, standard_error, posterior_beta, 
                                                                                                                              selector_logits, selected, critic_mean, critic_var)
            
            
            running_latent_loss += latent_loss.detach()
            running_spike_loss += torch.mean(kl_spike.detach())
            running_slab_loss += torch.mean(kl_slab.detach())
            running_recon_loss += torch.mean(beta_hat_loss.detach())
            latent_loss.backward()
            #selector_optimizer.step()
            only_latent_opt.step()
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
                ax1.legend(loc='lower right')
                ax1.set_ylabel('Beta')
                ax1.set_xlabel('Locus')
                
                ax3 = fig.add_subplot(gs[1,0])
                ax3.plot(mean_estimate.mean(axis=0)[:25],'--', c='r', label='Reconstructed Beta')
                ax3.plot(beta.mean(axis=0)[:25],c='b', label='Observed Beta')
                ax3.legend(loc='lower right')
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
    #run_model('30690NA_gene_effects')
    run_model_test()