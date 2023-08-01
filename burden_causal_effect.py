import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nflows import distributions as distributions_
from nflows import flows, transforms
from burden_selector import BurdenSelector
from nflows.nn.nets import ResidualNet
from functools import partial
from pyro.nn import DenseNN
from pyro.distributions import transforms, RelaxedBernoulliStraightThrough

class AffineTransform(transforms.AffineTransform):
    """Trainable version of an Affine transform. This can be used to get diagonal
    Gaussian approximations."""

    __doc__ += transforms.AffineTransform.__doc__

    def parameters(self):
        self.loc.requires_grad_(True)
        self.scale.requires_grad_(True)
        yield self.loc
        yield self.scale

    def with_cache(self, cache_size=1):
        if self._cache_size == cache_size:
            return self
        return AffineTransform(self.loc, self.scale, cache_size=cache_size)

class BurdenEncoder(nn.Module):
    def __init__(self, input_sz, hidden_szs, latent_sz, num_layers):
        super().__init__()
        self.input_sz = input_sz # Max number of snps 
        self.hidden_szs = hidden_szs 
        self.n_layer = num_layers
        self.latent_sz = latent_sz
        self.encoder = nn.Sequential()

     

        # Build network
        self.encoder.append(nn.Linear(self.input_sz, self.hidden_szs))
        # setup the three linear transformations used
        for _ in range(self.n_layer - 2):
            self.encoder.append(nn.Linear(self.hidden_szs, self.hidden_szs))
            self.encoder.append(nn.SiLU())
        
        self.mean_linear = nn.Linear(self.hidden_szs, self.latent_sz)
        self.log_var_linear = nn.Linear(self.hidden_szs, self.latent_sz)
        self.spike_linear = nn.Linear(self.hidden_szs, self.latent_sz)


    def forward(self, x):

        z = self.encoder(x)
        mean = self.mean_linear(z)
        log_var = torch.log(torch.nn.functional.softplus(self.log_var_linear(z)))
        spike = torch.nn.functional.sigmoid(self.spike_linear(z)) # create probabilities
        if torch.any(torch.isnan(log_var)): 
            print("WTF")
        if torch.any(torch.isnan(mean)): 
            print("WTF")
        return z, mean, log_var, spike
        #return z, mean, log_var

class BurdenDecoder(nn.Module):
    def __init__(self, input_sz, hidden_szs, latent_sz, num_layers):
        super().__init__()
        self.input_sz = input_sz # Max number of snps 
        self.hidden_szs = hidden_szs 
        self.n_layer = num_layers
        self.latent_sz = latent_sz
        self.decoder = nn.Sequential()
        
        # setup the three linear transformations used
        self.decoder.append(nn.Linear(self.latent_sz, self.hidden_szs))
        for _ in range(self.n_layer - 2):
            self.decoder.append(nn.Linear(self.hidden_szs, self.hidden_szs))
            self.decoder.append(nn.SiLU())
        
        self.mean_linear = nn.Linear(self.hidden_szs, self.input_sz)
        self.log_var_linear = nn.Linear(self.hidden_szs, self.input_sz)


    def forward(self, x):
        
        x_hat = self.decoder(x)
        mean = self.mean_linear(x_hat)
        log_var = torch.log(torch.nn.functional.softplus(self.log_var_linear(x_hat)))
        return mean, log_var
    

class BurdenCritic(nn.Module):
    def __init__(self, input_sz, hidden_szs, num_layers):
        super().__init__()
        self.input_sz = input_sz # Max number of snps 
        self.hidden_szs = hidden_szs 
        self.n_layer = num_layers
        self.encoder = nn.Sequential()

        # setup the three linear transformations used
        for _ in range(self.n_layer - 2):
            self.encoder.append(nn.Linear(self.input_sz, self.hidden_szs))
            self.encoder.append(nn.ReLU())
        
        self.encoder.append(nn.Linear(self.hidden_szs,self.input_sz))

    def forward(self, x):

        q = self.encoder(x)

        return q

class BurdenActor(nn.Module):
    def __init__(self, input_sz, hidden_szs, latent_sz, num_layers):
        super(BurdenActor, self).__init__()

        self.input_sz = input_sz # Max number of snps 
        self.hidden_szs = hidden_szs 
        self.n_layer = num_layers
        self.latent_sz = latent_sz
        self.encoder = nn.Sequential()

        # setup the three linear transformations used
        for _ in range(self.n_layer - 2):
            self.encoder.append(nn.Linear(self.input_sz, self.hidden_szs))
            self.encoder.append(nn.ReLU())
        
        self.encoder.append(nn.Linear(self.hidden_szs,self.latent_sz))

    def forward(self, state):

        a = self.encoder(state)
        a = torch.tanh(a) # create logits

        return a


class VariationalBurden(nn.Module):
    def __init__(self, num_variants: int=10, hidden_sz: int=5, latent_sz: int=1, batch_size: int=25, lambda_param: float=0.1,
                 num_transforms: int=1, num_blocks: int=1, dropout_probability: float=0.1, permute_flow: bool=False, use_batch_norm: bool=False, num_bins: int=2,
                 tail_bound: float=8.0, device: bool=False):
        super().__init__()
        
        self.input_sz = num_variants
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.num_transforms = num_transforms
        self.num_blocks = num_blocks
        self.dropout_probability = dropout_probability
        self.use_batch_norm = use_batch_norm
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.permute = permute_flow
        self.batch_norm = use_batch_norm
        self.lamda_hyper_param = lambda_param
        self.device = device
        self.alpha= 1/num_variants # 
        self.temperature = torch.tensor(.2)
        self.batch_size = batch_size
        self.reinforcement_learning = True

        # Build pyro based priors
        self.latent_prior, self.latent_net = self.burden_critic_baseline_nsf_pyro()

        # Build Encoder
        self.encoder = BurdenEncoder(self.input_sz, self.hidden_sz, self.latent_sz, num_layers=1)
        
        # Build decoder
        self.decoder = BurdenDecoder(self.input_sz, self.hidden_sz, self.latent_sz, num_layers=1)

        # Build Critic
        self.critic_net = BurdenCritic(self.input_sz, self.hidden_sz, num_layers=3)

        #Build Selector network/Actor Network
        self.selector_network = self.burden_actor()

    
    def gaussian_entropy(self, logvar):
        # multivariate gaussian entropy

        const = 0.5 * (logvar.size(1)) * (1. + torch.log(torch.tensor(torch.pi) * 2)) # D/2 * (1+ log(2*pi) where D is size of covariance matrix, or number of columns
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const # guassian has a diagonal covariance, so determinant is just sum of diagonals
        return ent
    
    # Reconstruction + KL divergence of normalizing flow losses summed over all elements of batch
    def critic_loss_function(self, x, true_beta, selectors, standard_error, mask=None):
        selected = self.selector_network.bernoulli_sampling(selectors)
        if mask is not None:
            # element-wise product, for selected variants/features, but mask out variants that never had an effect size
            true_beta_selected = torch.mul(true_beta, selected)
            true_beta_selected = torch.mul(true_beta_selected, mask) 
        else:
            true_beta_selected = torch.mul(true_beta, selected) # element-wise product, for selected variants/features

        # Loss function for critic
        loss_fn_critic = nn.GaussianNLLLoss(reduction='none')
        critic_loss = torch.sum(loss_fn_critic(true_beta_selected, x, standard_error),axis=1) # shape batch x 1
        critic_prior_loss = self.critic_prior.log_prob(true_beta) # shape batch x 1



        return -(torch.mean(critic_loss-critic_prior_loss)), selected 

    # Reconstruction + KL divergence of normalizing flow losses summed over all elements of batch
    def critic_loss_function_with_vae(self, x, standard_error, posterior_beta, true_beta_mean, true_beta_var, selectors):
        selected = self.selector_network.bernoulli_sampling(selectors)
        posterior_beta_selected = torch.mul(posterior_beta, selected) # element-wise product, for selected variants/features

        # Loss function for critic
        # Calculate entropy of posterior q(B|estimated beta, se)
        entropy = self.gaussian_entropy(true_beta_var) 

        # Calculate log_prob of seeing posterior under prior # EQφ(z|x)[log Pψ(z)] ≈ sum [ log Pψ(µ + eps*σ) ]
        critic_true_beta = self.critic_prior.log_prob(posterior_beta)

        # calculate reconstruction error
        mean_estimate, log_var_estimate = self.decoder(posterior_beta_selected)
        beta_hat = mean_estimate + torch.randn_like(log_var_estimate) * torch.exp(0.5*log_var_estimate)
        mean_mse_fn = nn.MSELoss(reduction='none')        
        mean_loss = torch.sum(mean_mse_fn(beta_hat, x), axis=1)
        

        total_loss = torch.mean(mean_loss + entropy - critic_true_beta)

        return  -total_loss, selected 

    def latent_variable_update(self, x, standard_error, posterior_beta, true_beta_mean, true_beta_var):

        # Loss function for critic
        # Calculate entropy of posterior q(B|estimated beta, se)
        #entropy = self.gaussian_entropy(true_beta_var) 

        # Calculate log_prob of seeing posterior under prior # EQφ(z|x)[log Pψ(z)] ≈ sum [ log Pψ(µ + eps*σ) ]
        #critic_true_beta = self.critic_prior.log_prob(posterior_beta)

        #This +KL(Q(z|x)||p(z)) -- note the positive
        kl_test = 0.5 * torch.sum(true_beta_mean.pow(2) + true_beta_var.exp() - 1 - true_beta_var, axis=1, keepdim=True) 

        # calculate reconstruction error
        #mean_estimate, log_var_estimate = self.decoder(posterior_beta)
        #beta_hat = mean_estimate + torch.randn_like(log_var_estimate) * torch.exp(0.5*log_var_estimate)
        #mean_mse_fn = nn.MSELoss(reduction='none')        
        #mean_loss = torch.sum(mean_mse_fn(beta_hat, x), axis=1)
        loss_fn = nn.GaussianNLLLoss(reduction='none') 
        #beta_hat = torch.sum(loss_fn(mean_estimate, x, log_var_estimate.exp()),axis=1, keepdim=True)
        beta_hat = torch.sum(loss_fn(true_beta_mean, x, true_beta_var.exp()),axis=1, keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            beta_hat2 =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            
        #total_loss = torch.mean(mean_loss + entropy + critic_true_beta)

        # This is the simple loss function error we are trying to minimize, mathematically we want to maximize, max(L) = E[P(X|z)] - KLD(q||p)
        # But autograd/pytorch minimizes, so we intead minmize min(L) = -max(L) = -E[P(X|z)] + KLD(q||p)
        total_loss = torch.mean(beta_hat+kl_test,axis=0 )

        return total_loss, beta_hat2 
    
    def latent_variable_update_with_spike(self, x, standard_error, posterior_beta, spike, selected, true_beta_mean, true_beta_var):

        self.alpha=0.5
     

        #This +KL(Q(z|x)||p(z)) -- note the positive 
        #kl_part_1 = 0.5 * torch.sum(torch.mul(spike,true_beta_mean.pow(2) + true_beta_var.exp() - 1 - true_beta_var), axis=1, keepdim=True)
        kl_part_1 = -true_beta_var - 0.5
        kl_part_2 = (true_beta_mean.pow(2) + true_beta_var.exp())*0.5
        kl_slab_loss = torch.sum(torch.mul(spike, kl_part_1+kl_part_2),axis=1,keepdim=True)
        

        # Spike loss
        spike_kl_part_1 = (1 - spike).mul(torch.log((1 - spike)) / (1 - self.alpha))
        spike_kl_part_2 = spike.mul(torch.log(spike/self.alpha))
        if torch.any(torch.isnan(kl_slab_loss)):
            print("WTF is KL test or posterior/gaussian")
        if torch.any(torch.isnan(spike_kl_part_1)):
            print("Spike 1 WTF")
            print("Logspike is: {}".format(spike))
        if torch.any(torch.isnan(spike_kl_part_2)):
            print("Spike 2 WTF")
        
        
        spike_kl = torch.sum(spike_kl_part_1+spike_kl_part_2,axis=1, keepdim=True)
        
        
        # calculate recostruction error
        loss_fn = nn.GaussianNLLLoss(reduction='none') 
        beta_hat_loss = torch.sum(loss_fn(true_beta_mean*selected, x, (true_beta_var*selected).exp()),axis=1, keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            true_beta =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            

        # This is the simple loss function error we are trying to minimize, mathematically we want to maximize, max(L) = E[P(X|z)] - KLD(q||p)
        # But autograd/pytorch minimizes, so we intead minmize min(L) = -max(L) = -E[P(X|z)] + KLD(q||p)
        total_loss = torch.mean(beta_hat_loss+kl_slab_loss+spike_kl,axis=0 )

        return total_loss, beta_hat_loss.detach(), kl_slab_loss.detach(), spike_kl.detach(), posterior_beta.detach(), true_beta.detach()
    
    def latent_variable_update_with_spike_and_decoder(self, x, mask, standard_error, posterior_beta, logspike, selected, true_beta_mean, true_beta_var):
        """_summary_

        Args:
            x (_type_): _description_
            standard_error (_type_): _description_
            posterior_beta (_type_): _description_
            spike (_type_): logits of concrete distribution that approximates a bernoulli
            selected (_type_): log(Y), where Y ~ (1,0), Y ~ gumbel_sigmoid(spike)
            true_beta_mean (_type_): _description_
            true_beta_var (_type_): _description_

        Returns:
            _type_: _description_
        """        
        #This +KL(Q(z|x)||p(z)) -- note the positive 
        #kl_part_1 = 0.5 * torch.sum(torch.mul(spike,true_beta_mean.pow(2) + true_beta_var.exp() - 1 - true_beta_var), axis=1, keepdim=True)
        kl_part_1 = -0.5*true_beta_var - 0.5
        kl_part_2 = (true_beta_mean.pow(2) + true_beta_var.exp())*0.5
        spike = torch.mul(logspike.exp(), mask) + 1e-6
        spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
        #spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6) 
        #spike = torch.mul(mask, spike) + 1e-12
        kl_slab_loss = torch.sum(torch.mul(spike, kl_part_1+kl_part_2),axis=1,keepdim=True)
        
        #mask_prior = torch.sum(mask,axis=1,keepdim=True)*mask
        #mask_prior[:,0] += 1
        #mask_prior = torch.nan_to_num(torch.pow(mask_prior, torch.tensor(-1)),nan=0.0,posinf=0.0,neginf=0.0)
        #prior_probs = torch.ones_like(logspike)*mask_prior + 1e-12
        prior_probs = torch.ones_like(logspike)*mask*0.5 + 1e-6
        spike_kl_part_1 = (1 - spike).mul(torch.log((1 - spike) / (1 - prior_probs)))
        spike_kl_part_2 = spike.mul(torch.log(spike/prior_probs))
        # Spike loss -- this is the analytical divergence between two binary relaxed bernoulli/concrete variables
        #prior_probs = torch.log(torch.ones_like(selected)*(1/2)*mask + 1e-12)
        #q_z_x = self.concrete_log_density(selected, spike, temperature=self.temperature)
        #p_z = self.concrete_log_density(selected, prior_probs, temperature=self.temperature )
        

        #q_sampler = RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=spike)
        q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, probs=spike)
        #p_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, logits=prior_probs)
        q_z_x_samples = torch.mul(q_sampler.rsample(),mask)
        spike_kl = torch.sum(spike_kl_part_1+spike_kl_part_2,axis=1, keepdim=True)
        
        # Find selected features
        #eta = torch.rand_like(logspike)
        #u = torch.log(eta) - torch.log(1 - eta)
        #selected = torch.sigmoid((u + logspike) / self.temperature)
        #selected = torch.mul(selected, mask)

        # Get reconstruction
        posterior_beta = torch.mul(q_z_x_samples,posterior_beta)
        mean_estimate = self.decoder(posterior_beta)
        mean_estimate = torch.mul(mean_estimate,mask)

        # calculate reconstruction error      

        loss_fn = nn.GaussianNLLLoss(reduction='none') 
        beta_hat_loss = torch.sum(loss_fn(mean_estimate, x, standard_error),axis=1, keepdim=True)

        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            true_beta =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            beta_hat_sampler =  torch.distributions.normal.Normal(mean_estimate.detach(),torch.sqrt(standard_error+1e-6))
            recon = beta_hat_sampler.sample()

        #This is the simple loss function error we are trying to minimize, mathematically we want to maximize, max(L) = E[P(X|z)] - KLD(q||p)
        # But autograd/pytorch minimizes, so we intead minmize min(L) = -max(L) = -E[P(X|z)] + KLD(q||p)
        total_loss = torch.mean(beta_hat_loss+0.9*kl_slab_loss+0.9*spike_kl,axis=0 )

        return total_loss, beta_hat_loss.detach(), kl_slab_loss.detach(), spike_kl.detach(), posterior_beta.detach(), true_beta.detach(), recon.detach()
    
    def iwae_latent_variable_update_with_spike_and_decoder(self, x, mask, standard_error, posterior_beta, posterior_dist, logit_spike, selected, 
                                                           true_beta_mean, true_beta_var, num_samples, batch_size):
        """Importance weighted elbo

        Args:
            x (_type_): _description_
            standard_error (_type_): _description_
            posterior_beta (_type_): _description_
            true_beta_mean (_type_): _description_
            true_beta_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        spike = torch.mul(logit_spike.exp(), mask) + 1e-6
        spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
        #IW-ELBO is:
        #E[LOG[P(X|z)] - LOG[KL(Q||P)]] Note that the log is inside the expectation

        # Find q(z|x)
        log_Q_e_Fx = posterior_dist.log_prob(posterior_beta)
        
        # Find p(z) -- Gaussian unit variance
        mu_prior = torch.zeros_like(posterior_beta)
        std_prior = torch.ones_like(posterior_beta)
        p_z = torch.distributions.Normal(loc=mu_prior, scale=std_prior)
        p_z_log_prob = p_z.log_prob(posterior_beta)

        # Find p(z) - q(z|x)

        iw_qz_pz = p_z_log_prob - log_Q_e_Fx

        # Find pi*[p(z) - q(z|x)]
        slab_loss = torch.sum(torch.mul(spike, iw_qz_pz),axis=-1, keepdim=True)

        # find q(pi | x)
        #q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, logits=logit_spike)
        q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, probs=spike)
        pi_samples = q_sampler.rsample()
        log_Q_e_Fpi = q_sampler.log_prob(pi_samples)

        # find p(pi)
        mask_prior = torch.sum(mask,axis=-1,keepdim=True)*mask
        mask_prior = torch.nan_to_num(torch.pow(mask_prior, torch.tensor(-1)),nan=1.0,posinf=0.0,neginf=0.0) # create uniform probs except where 0
        prior_probs = torch.ones_like(logit_spike)*mask_prior
        p_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, probs=prior_probs)
        log_P_d_Fpi = p_sampler.log_prob(pi_samples)

        # find p(pi) - q(pi|x)

        iw_qpi_ppi = log_P_d_Fpi - log_Q_e_Fpi
        spike_loss = torch.sum(iw_qpi_ppi, axis=-1, keepdim=True)

        # Find p(x|z)
        posterior_beta = torch.mul(pi_samples,posterior_beta)
        mean_estimate, log_var_estimate = self.decoder(posterior_beta)
        mean_estimate = torch.mul(mean_estimate, mask)
        log_var_estimate = torch.mul(log_var_estimate, mask)
        beta_hat_sampler = torch.distributions.Normal(loc=mean_estimate, scale=log_var_estimate.mul(0.5).exp_()).log_prob(x)
        beta_hat_loss = torch.sum(beta_hat_sampler, axis=-1,keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            true_beta =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            beta_hat_sampler_2 =  torch.distributions.Normal(loc=mean_estimate, scale=log_var_estimate.mul(0.5).exp_())
            recon = beta_hat_sampler_2.sample()

        log_weights = (beta_hat_loss + slab_loss + spike_loss).squeeze(-1)
        log_weights =  log_weights.reshape(batch_size, num_samples) # Shape of [batch_size, K]
        with torch.no_grad():
            log_w_tilde = log_weights - torch.logsumexp(log_weights,dim=-1,keepdim=True) # normalize weights over K-samples
            w_tilde = log_w_tilde.exp()
            posterior_beta.register_hook(lambda grad: w_tilde.reshape(batch_size*num_samples).unsqueeze(-1) * grad.reshape(batch_size*num_samples,-1))
        sum_over_iwae_samples = torch.sum(w_tilde*log_weights,axis=-1) # Sum over K-samples
        total_loss = -torch.mean(sum_over_iwae_samples)

        return total_loss, beta_hat_loss.detach(), slab_loss.detach(), spike_loss.detach(), posterior_beta.detach(), true_beta.detach(), recon.detach(), pi_samples, 0

    
    
    def latent_variable_update_with_flow(self, x, standard_error, posterior_beta, true_beta_mean, true_beta_var):

        # Loss function for critic
        # Calculate entropy of posterior q(B|estimated beta, se)
        entropy = self.gaussian_entropy(true_beta_var) 

        # Calculate log_prob of seeing posterior under prior # EQφ(z|x)[log Pψ(z)] ≈ sum [ log Pψ(µ + eps*σ) ]
        critic_true_beta = self.latent_prior.log_prob(posterior_beta)


        # calculate reconstruction error
        #mean_estimate, log_var_estimate = self.decoder(posterior_beta)
        #beta_hat = mean_estimate + torch.randn_like(log_var_estimate) * torch.exp(0.5*log_var_estimate)
        #mean_mse_fn = nn.MSELoss(reduction='none')        
        #mean_loss = torch.sum(mean_mse_fn(beta_hat, x), axis=1)
        loss_fn = nn.GaussianNLLLoss(reduction='none') 
        beta_hat = torch.sum(loss_fn(true_beta_mean, x, true_beta_var.exp()),axis=1, keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            beta_hat2 =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            
        #total_loss = torch.mean(mean_loss + entropy + critic_true_beta)

        # This is the simple loss function error we are trying to minimize, mathematically we want to maximize, max(L) = E[P(X|z)] - KLD(q||p)
        # But autograd/pytorch minimizes, so we intead minmize min(L) = -max(L) = -E[P(X|z)] + KLD(q||p)
        total_loss = torch.mean(beta_hat - entropy - critic_true_beta)

        return total_loss, beta_hat2 
    
    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        if self.latent_prior._validate_args:
            self.latent_prior._validate_sample(value)
        event_dim = len(self.latent_prior.event_shape)
        log_prob = 0.0
        y = value
        transforms = self.latent_prior.transforms
        for transform in reversed(transforms):
            x = transform.inv(y)
            event_dim += transform.domain.event_dim - transform.codomain.event_dim # should remain constant at 1
            det = transform.log_abs_det_jacobian(x, y) # need to change when using simple affine-gaussian to .sum(-1)

            log_prob = log_prob - det
            y = x

        base_log_prob = self.latent_prior.base_dist.base_dist.log_prob(y)
                                             
        return base_log_prob, log_prob
    
    def latent_variable_update_with_flow_and_decoder(self, x, mask, standard_error, posterior_beta, logspike, selected, true_beta_mean, true_beta_var):
        """_summary_

        Args:
            x (_type_): _description_
            standard_error (_type_): _description_
            posterior_beta (_type_): _description_
            true_beta_mean (_type_): _description_
            true_beta_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        #ELBO is:
        #E[P(X|z)] - E[KL(Q||P)]
        # Since we are using a flow prior
        # KL(Q||P) = -1*(E[f^-1(P(z))] + H(Q)) where H is entropy
        # E[P(X|z)] + (E[f^-1(P(z))] + H(Q))
        # To add spike this becomes:
        #E[P(X|z)] + spike*((E[f^-1(P(z))] + H(Q))
         #This +KL(Q(z|x)||p(z)) -- note the positive 
        # Calculate log_prob of seeing posterior under prior # EQφ(z|x)[log Pψ(z)] ≈ sum [ log Pψ(µ + eps*σ) ]
        spike = torch.mul(logspike.exp(), mask)
        spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
   
        entropy = self.gaussian_entropy(true_beta_var)
        
        if torch.any(torch.isnan(posterior_beta)):
            print("WTF is posterior beta in flow and decoder")
        log_prob, det = self.log_prob(posterior_beta)
 
        total_log_prob = log_prob + torch.diagonal_scatter(torch.zeros_like(log_prob),det) + torch.diagonal_scatter(torch.zeros_like(log_prob),entropy)
        weighted_log_prob = torch.mul(spike, total_log_prob)

        kl_slab_loss = torch.sum(weighted_log_prob,axis=1,keepdim=True)
        

        # Spike loss
        prior_probs = torch.ones_like(logspike)*mask*0.15 + 1e-6
        spike_kl_part_1 = (1 - spike).mul(torch.log((1 - spike) / (1 - prior_probs)))
        spike_kl_part_2 = spike.mul(torch.log(spike/prior_probs))
        spike_kl = torch.sum(spike_kl_part_1+spike_kl_part_2,axis=1, keepdim=True)

        # Selector net/sampler
        q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, logits=logspike)
        # Calculated which columns are selected
        #q_sampler = RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=spike)        
        q_z_x_samples = torch.clamp(torch.mul(q_sampler.rsample(),mask),1e-6,1)
        # calculate reconstruction error
        posterior_beta = torch.mul(q_z_x_samples,posterior_beta)
        mean_estimate = self.decoder(posterior_beta)
        mean_estimate = torch.mul(mean_estimate,mask)
        loss_fn = nn.GaussianNLLLoss(reduction='none') 
        beta_hat_loss = torch.sum(loss_fn(mean_estimate, x, standard_error),axis=1, keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            true_beta =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            beta_hat_sampler =  torch.distributions.normal.Normal(mean_estimate,torch.sqrt(standard_error+1e-6))
            recon = beta_hat_sampler.sample()

        # total loss = ELBO = Recon - (KL_SLAB + KL_SPIKE)
        # total loss = ELBO = Recon - [(-E[f^-1(z)] - H) + KL_SPIKE)
        # total loss = ELBO = Recon + E[f^-1(z)] + H - KL_SPIKE
        # need to minimize so -ELBO
        # total loss = -ELBO = Negative_log_likli - (E[f^-1(z)] + H) + KL_SPIKE

        total_loss = torch.mean(beta_hat_loss - kl_slab_loss + 10*spike_kl,axis=0 )
        #total_loss = torch.mean(-kl_slab_loss,axis=0 )

        return total_loss, beta_hat_loss.detach(), kl_slab_loss.detach(), 10*spike_kl.detach(), posterior_beta.detach(), true_beta.detach(), recon.detach(), q_z_x_samples, self.latent_prior
    
    def iwae_latent_variable_update_with_flow_and_decoder(self, x, mask, standard_error, posterior_beta, posterior_dist, logit_spike, selected, 
                                                           true_beta_mean, true_beta_var, num_samples, batch_size):
        """Importance weighted elbo

        Args:
            x (_type_): _description_
            standard_error (_type_): _description_
            posterior_beta (_type_): _description_
            true_beta_mean (_type_): _description_
            true_beta_var (_type_): _description_

        Returns:
            _type_: _description_
        """
        spike = torch.mul(logit_spike.exp(), mask) + 1e-6
        spike = torch.clamp(spike, 1e-6, 1.0-1e-6)
        #IW-ELBO is:
        #E[LOG[P(X|z)] - LOG[KL(Q||P)]] Note that the log is inside the expectation

        # Find q(z|x)
        log_Q_e_Fx = posterior_dist.log_prob(posterior_beta)
        
        # Find p(z)
        p_z_log_prob, p_z_det = self.log_prob(posterior_beta)
        p_z_total_log_prob = p_z_log_prob + torch.diagonal_scatter(torch.zeros_like(p_z_log_prob),p_z_det)

        # Find p(z) - q(z|x)

        iw_qz_pz = p_z_total_log_prob - log_Q_e_Fx

        # Find pi*[p(z) - q(z|x)]
        slab_loss = torch.sum(torch.mul(spike, iw_qz_pz),axis=-1, keepdim=True)

        # find q(pi | x)
        #q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, logits=logit_spike)
        q_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, probs=spike)
        pi_samples = q_sampler.rsample()
        log_Q_e_Fpi = q_sampler.log_prob(pi_samples)

        # find p(pi)
        mask_prior = torch.sum(mask,axis=-1,keepdim=True)*mask
        mask_prior = torch.nan_to_num(torch.pow(mask_prior, torch.tensor(-1)),nan=1.0,posinf=0.0,neginf=0.0) # create uniform probs except where 0
        prior_probs = torch.ones_like(logit_spike)*mask_prior
        p_sampler = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature=self.temperature, probs=prior_probs)
        log_P_d_Fpi = p_sampler.log_prob(pi_samples)

        # find p(pi) - q(pi|x)

        iw_qpi_ppi = log_P_d_Fpi - log_Q_e_Fpi
        spike_loss = torch.sum(iw_qpi_ppi, axis=-1, keepdim=True)

        # Find p(x|z)
        posterior_beta = torch.mul(pi_samples,posterior_beta)
        mean_estimate, log_var_estimate = self.decoder(posterior_beta)
        mean_estimate = torch.mul(mean_estimate, mask)
        log_var_estimate = torch.mul(log_var_estimate, mask)
        beta_hat_sampler = torch.distributions.Normal(loc=mean_estimate, scale=log_var_estimate.mul(0.5).exp_()).log_prob(x)
        beta_hat_loss = torch.sum(beta_hat_sampler, axis=-1,keepdim=True)
        
        with torch.no_grad():
            eps = torch.randn_like(true_beta_var)
            true_beta =  true_beta_mean+eps*torch.exp(0.5*true_beta_var)
            beta_hat_sampler_2 =  torch.distributions.Normal(loc=mean_estimate, scale=log_var_estimate.mul(0.5).exp_())
            recon = beta_hat_sampler_2.sample()

        log_weights = (beta_hat_loss + slab_loss + spike_loss).squeeze(-1)
        log_weights =  log_weights.reshape(batch_size, num_samples) # Shape of [batch_size, K]
        with torch.no_grad():
            log_w_tilde = log_weights - torch.logsumexp(log_weights,dim=-1,keepdim=True) # normalize weights over K-samples
            w_tilde = log_w_tilde.exp()
            posterior_beta.register_hook(lambda grad: w_tilde.reshape(batch_size*num_samples).unsqueeze(-1) * grad.reshape(batch_size*num_samples,-1))
        sum_over_iwae_samples = torch.sum(w_tilde*log_weights,axis=-1) # Sum over K-samples
        total_loss = -torch.mean(sum_over_iwae_samples)


        return total_loss, beta_hat_loss.detach(), slab_loss.detach(), spike_loss.detach(), posterior_beta.detach(), true_beta.detach(), recon.detach(), pi_samples, self.latent_prior


    def baseline_loss_function(self, x, true_beta, standard_error, mask=None):

        # Loss function for baseline
        if mask is not None:
            # element-wise product, for selected variants/features, but mask out variants that never had an effect size
            true_beta = torch.mul(true_beta, mask) 
        
        loss_fn_baseline = nn.GaussianNLLLoss(reduction='none')
        baseline_loss = torch.sum(loss_fn_baseline(true_beta, x, standard_error),axis=1) # shape batch x 1
        baseline_prior_loss = self.baseline_prior.log_prob(true_beta) # shape batch x 1


        return -(torch.mean(baseline_loss-baseline_prior_loss)) 

    def selector_loss(self, critic_loss, baseline_loss, selectors, selector_logits):
        """Calcuates the loss function or the policy gradient for choosing the selected variants/features

        Args:
            critic_loss (_type_): this is loss of reconstruction the estimated effect sizes given the true betas after selection
            baseline_loss (_type_): this is loss of reconstruction the estimated effect sizes given the true betas without selection
            selectors (_type_): These are the selected varaints is a matrix of 1 and 0
            selector_logits (_type_): These are the logits, or the output of the selector network
        """        

        Reward = -(critic_loss - baseline_loss)
        policy_gradient = Reward*torch.sum(selectors * torch.log(selector_logits + 1e-8) + (1-selectors) * torch.log(1-selector_logits + 1e-8), axis = 1) - \
                                    self.lamda_hyper_param * torch.mean(selector_logits, axis = 1)
        
        return -torch.mean(policy_gradient,axis=0) # mean over batch

    def critic_loss(self, x, aggregated_posterior, selector_logits):


        selectors = self.selector_network.bernoulli_sampling(selector_logits)        
        true_beta_selected = torch.mul(aggregated_posterior, selectors)
        mean_estimate = self.critic_net(true_beta_selected)

        loss_fn = nn.MSELoss(reduction='none') 
        loss = torch.sum(loss_fn(mean_estimate, x),axis=1, keepdim=True)
        return torch.mean(loss,axis=0), selectors, mean_estimate.detach()

    def selector_loss_vae(self, x, critic_loss, selectors, selector_logits, mask):
        """Calcuates the loss function or the policy gradient for choosing the selected variants/features

        Args:
            critic_loss (_type_): this is loss of reconstruction the estimated effect sizes given the true betas after selection
            baseline_loss (_type_): this is loss of reconstruction the estimated effect sizes given the true betas without selection
            selectors (_type_): These are the selected varaints is a matrix of 1 and 0
            selector_logits (_type_): These are the logits, or the output of the selector network
        """
        loss = nn.BCEWithLogitsLoss(reduction='none')
        Reward = -critic_loss
        regularizer = loss(selector_logits, mask)
        policy_gradient = Reward*torch.sum(selectors * torch.log(selector_logits + 1e-8) + (1-selectors) * torch.log(1-selector_logits + 1e-8), axis = 1) - \
            self.lamda_hyper_param * torch.sum(regularizer, axis = 1)
        
    
        
        return -torch.mean(policy_gradient,axis=0) # mean over batch
    
    def concrete_log_density(self, selected, selectors, temperature):
        """
        Calculate log density of concrete variables

        Args:
            selected (_type_): _description_
            selectors (_type_): _description_
            temperature (_type_): _description_
        """    
        arg = (-selected.exp()*temperature + selectors) # -temperature*y + log(alpha) Equation 26 Appendix C https://arxiv.org/pdf/1611.00712.pdf
        log_one_plus_exp = torch.logaddexp(torch.tensor(0),arg) #(log(1+ exp(-temperature*y + log(alpha)))
        log_density = torch.log(temperature) - temperature*selected.exp() + (-2*log_one_plus_exp)

        return log_density

    def burden_actor(self):
        """ 
            Build actor, find selected features.
        """
        if self.reinforcement_learning:
            return BurdenActor(self.input_sz, self.hidden_sz, self.latent_sz, num_layers=2)
        else:
            return BurdenSelector(self.input_sz, self.hidden_sz, self.latent_sz, 0, num_layers=2)
    
    def burden_critic_baseline_nsf_pyro(self, base=None):
        dim = self.input_sz
        split_dim = self.input_sz // 2
        hidden_dims = [2*dim, 2 *dim]
        nonlinearity = nn.SiLU()
        param_dims = [
            (dim - split_dim) * self.num_bins,
            (dim - split_dim) * self.num_bins,
            (dim - split_dim) * (self.num_bins - 1),
            (dim - split_dim) * self.num_bins,
        ]
        #base_dist = torch.distributions.MultivariateNormal(torch.zeros(self.input_sz), torch.eye(self.input_sz))
        if base is not None:
            base_dist = base       
        else:
            base_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(self.input_sz),
                torch.ones(self.input_sz),),
                1,)
        hypernet = DenseNN(split_dim, hidden_dims, param_dims, nonlinearity=nonlinearity)
        if self.device:
            hypernet.to('cuda')
        transform = transforms.SplineCoupling(dim, split_dim, hypernet, self.num_bins, self.tail_bound)

        flows = []
        for i in range(self.num_transforms):
            flows.append(transform.with_cache())
            if self.permute and i < self.num_transforms - 1:
                permutation = torch.randperm(dim)
                flows.append(transforms.Permute(permutation))
            if self.batch_norm and i < self.num_transforms - 1:
                bn = transforms.BatchNorm(dim)
                flows.append(bn)
        
        a_dist = torch.distributions.TransformedDistribution(base_dist, flows)
        
        return a_dist, transforms.ComposeTransformModule(flows).with_cache()
        #return TransformedDistribution(base_dist, flows)
    
    def burden_critic_baseline_affine_diag_pyro(self, base=None):
        dim = self.input_sz

        #base_dist = torch.distributions.MultivariateNormal(torch.zeros(self.input_sz), torch.eye(self.input_sz))
        if base is not None:
            base_dist = base       
        else:
            base_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(self.input_sz),
                torch.ones(self.input_sz),),
                1,)

        transform = AffineTransform(torch.zeros(self.input_sz), torch.ones(self.input_sz),event_dim=1)

        flows = []
        for i in range(self.num_transforms):
            flows.append(transform.with_cache())
            if self.permute and i < self.num_transforms - 1:
                permutation = torch.randperm(dim)
                flows.append(transforms.Permute(permutation))
            if self.batch_norm and i < self.num_transforms - 1:
                bn = transforms.BatchNorm(dim)
                flows.append(bn)
        
        a_dist = torch.distributions.TransformedDistribution(base_dist, flows)
        
        return a_dist, transforms.ComposeTransformModule(flows).with_cache()

    def burden_critic_baseline_affine_coupling_pyro(self, base=None):
        dim = self.input_sz
        split_dim = self.input_sz // 2
        hidden_dims = [2 * dim , 2 * dim]
        param_dims = [dim-split_dim, dim-split_dim]
        nonlinearity = nn.ReLU()
        log_scale_min_clip=-3.0
        log_scale_max_clip=3.0
  
        #base_dist = torch.distributions.MultivariateNormal(torch.zeros(self.input_sz), torch.eye(self.input_sz))
        if base is not None:
            base_dist = base       
        else:
            base_dist = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros(self.input_sz),
                torch.ones(self.input_sz),),
                1,)
        hypernet = DenseNN(split_dim, hidden_dims, param_dims, nonlinearity=nonlinearity)
        if self.device:
            hypernet.to('cuda')
        transform = transforms.AffineCoupling(split_dim, hypernet,log_scale_min_clip=log_scale_min_clip, log_scale_max_clip=log_scale_max_clip )

        flows = []
        for i in range(self.num_transforms):
            flows.append(transform.with_cache())
            if self.permute and i < self.num_transforms - 1:
                permutation = torch.randperm(dim)
                flows.append(transforms.Permute(permutation))
            if self.batch_norm and i < self.num_transforms - 1:
                bn = transforms.BatchNorm(dim)
                flows.append(bn)
        
        a_dist = torch.distributions.TransformedDistribution(base_dist, flows)
        
        return a_dist, transforms.ComposeTransformModule(flows).with_cache()
        #return TransformedDistribution(base_dist, flows)

    def forward(self, data):
        """Runs the model

        Args:
            betas (torch.floattensor): _description_
        """
        if self.reinforcement_learning:
            betas, mask = data

        else:
            betas, mask = data
                    
            _, critic_mean, log_critic_var, selectors = self.encoder(betas)
            #_, critic_mean, log_critic_var = self.encoder(betas)
            posterior_beta, posterior_dist = self.reparameterize(critic_mean, log_critic_var.exp())
            

            # Calculate q(B|estimated beta, se)
            
            #selectors =  self.selector_network([posterior_beta.detach(), mask]) # q(pi | true_beta)
            #selectors =  self.selector_network([betas, mask]) # q(pi | estimated_beta)
            #selectors = torch.nn.functional.relu(selectors)*-1.0 # logscale probabilities, log(pi)

            # doing this for logits
            
            #selectors = torch.nn.functional.sigmoid(selectors) # logits 
            #selectors = torch.mul(selectors,mask)
            #selectors = torch.where(selectors==0,0.5,selectors) # so that masked locations have logits of 0.5 which correspond to probability of 0.5
    
            # Calculate selected variants/features q(pi | X)
            selectors = torch.nn.functional.relu(selectors)*-1.0 # logscale probabilities, log(pi) 

            #posterior_beta = torch.mul(selected.exp(), posterior_beta)
            if torch.any(torch.isnan(posterior_beta)):
                print("WTF posterior beta in forward")

            # Calculate actor state
            #selectors = self.selector_network([betas, mask])

            #return (selectors, critic_true_beta, 0)
            #return (selectors, posterior_beta, critic_mean, log_critic_var)
            return (posterior_beta, posterior_dist, critic_mean, log_critic_var, selectors, 0)

    def reparameterize(self, mu, logvar):
        """Reparameterizes mean and standard from a neural network, this is too allow gradients to backpropogate

        Args:
            mu (_type_): _description_
            std (_type_): _description_

        Returns:
            _type_: _description_
        """        
        std = logvar.mul(0.5).exp_()
        qz_Gx_obs = torch.distributions.Normal(loc=mu, scale=std)
        # find z|x
        z_Gx = qz_Gx_obs.rsample() # make sure to use rsample for reparameterizeable
        return z_Gx, qz_Gx_obs


    def model_parameters(self):
        # Build pyro based priors
        #self.critic_prior, self.critic_net = self.burden_critic_baseline_nsf_pyro()
        #self.baseline_prior, self.baseline_net = self.burden_critic_baseline_nsf_pyro()

        #Build selector network
        #self.burden_actor()
        #model_param_list = {'critic': list(self.critic_net.parameters()), 'baseline': list(self.baseline_net.parameters()), 
        #                    'selector': list(self.selector_network.parameters())}

        #model_param_list = {'critic': list(self.critic_prior.parameters()), 'baseline': list(self.baseline_prior.parameters()), 
        #                    'selector': list(self.selector_network.parameters())}

        #model_param_list = {'encoder': list(self.encoder.parameters())+list(self.decoder.parameters()), 'prior': list(self.latent_net.parameters()), 'selector': list(self.selector_network.parameters())}
        model_param_list = {'encoder': list(self.encoder.parameters())+list(self.decoder.parameters()), 'prior': list(self.latent_net.parameters())}

        return model_param_list



        
        

  




    