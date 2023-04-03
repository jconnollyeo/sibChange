import numpy as np
import pymc3 as pm

def reinforce_probabilities(prob_array, sibling_indices, alpha):
    """
    Reinforce the probabilities in a 2D array using the probabilities of each pixel's siblings.
    
    Args:
        prob_array (ndarray): A 2D array of probabilities for each pixel in the original image.
        sibling_indices (ndarray): A 3D array of indices for each pixel's siblings in the original image.
        alpha (float): A weighting factor between 0 and 1 that determines the relative importance of the
            pixel's probability versus the average probability of its siblings.
    
    Returns:
        A new 2D array of reinforced probabilities.
    """
    
    prob_with_siblings = np.empty(shape=(SHP_i), dtype=float)
    sib_mask = ~np.isnan(SHP_i) * ~np.isnan(SHP_j)

    prob_with_siblings[sib_mask] = prob[SHP_i[sib_mask], SHP_j[sib_mask]]
    prob_with_siblings[~sib_mask] = np.nan

    siblings_prob = np.nanmean(prob_with_siblings, axis=0)

    reinforced_prob = alpha * prob_array + (1-alpha) * sibling_prob

    return reinforced_prob

def reinforce_probabilities_bayes(prob_array, sibling_indices, alpha, prior_alpha, prior_beta):
    """
    Reinforce the probabilities in a 2D array using Bayesian modeling that incorporates the probabilities
    of each pixel's siblings.
    
    Args:
        prob_array (ndarray): A 2D array of probabilities for each pixel in the original image.
        sibling_indices (ndarray): A 3D array of indices for each pixel's siblings in the original image.
        alpha (float): A weighting factor between 0 and 1 that determines the relative importance of the
            pixel's probability versus the average probability of its siblings.
        prior_alpha (float): The alpha parameter of the prior beta distribution for the pixel probabilities.
        prior_beta (float): The beta parameter of the prior beta distribution for the pixel probabilities.
    
    Returns:
        A new 2D array of reinforced probabilities.
    """
    # Define the prior distribution for the pixel probabilities
    prior_dist = pm.Beta.dist(alpha=prior_alpha, beta=prior_beta)
    
    # Initialize output array
    reinforced_probs = np.zeros_like(prob_array)
    
    # Loop over all pixels in the original image
    for i in range(prob_array.shape[0]):
        for j in range(prob_array.shape[1]):
            # Get the indices of the pixel's siblings
            sibling_idx = sibling_indices[i, j, :]
            
            # Define the prior distribution for the sibling probabilities
            prior_sibling_dist = pm.Beta.dist(alpha=prior_alpha, beta=prior_beta)
            
            with pm.Model() as model:
                # Define the likelihood function for the pixel probability
                likelihood = pm.Bernoulli('likelihood', p=prior_dist, observed=prob_array[i, j])
                
                # Define the likelihood function for the sibling probabilities
                sibling_likelihood = pm.Beta('sibling_likelihood', alpha=prior_sibling_dist.alpha + len(sibling_idx) * alpha,
                                             beta=prior_sibling_dist.beta + len(sibling_idx) * (1-alpha),
                                             observed=np.mean(prob_array[sibling_idx]))
                
                # Compute posterior distribution for the pixel probability
                posterior_dist = pm.Beta('posterior', alpha=prior_dist.alpha + likelihood.sum(),
                                         beta=prior_dist.beta + likelihood.size - likelihood.sum())
                
                # Sample from the posterior distribution
                trace = pm.sample(1000, tune=1000)
                
                # Compute the weighted average of the pixel and sibling probabilities
                reinforced_probs[i, j] = alpha * trace['posterior'].mean() + (1-alpha) * sibling_likelihood.distribution.mean()
    
    return reinforced_probs

def main():
    probabilities = np.load("nfs/a1/homes/py15jmc/bootstrap/2023/timeseries/probabilities_100_101_20230221_202044.npy")

    wdir = "/nfs/a1/homes/py15jmc/"

    SHP_i = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
    SHP_j = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

    r = reinforce_probabilities(prob_array, sibling_indices, alpha)

    fig, ax = plt.subplots(nrows = 1, ncols = 2)

    ax[0].pcolormesh(probabilities, vmin=0, vmax=1)
    ax[1].pcolormesh(r, vmin=0, vmax=1)

    plt.show()




# =============================================================

import pymc3 as pm
import numpy as np

def reinforce_probabilities_bayes(prob_array, SHP_i, SHP_j, alpha, beta):
    # Compute number of siblings for each pixel
    n_siblings = np.array([len(sibling_i) for sibling_i in SHP_i])
    # Add NaN padding to probability array to handle edge cases
    prob_with_padding = np.pad(prob_array, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
    # Create empty array for storing reinforced probabilities
    reinforced_probs = np.empty_like(prob_array)
    reinforced_probs[:] = np.nan
    # Loop over each pixel in the image
    for i in range(prob_array.shape[0]):
        for j in range(prob_array.shape[1]):
            # Get indices of sibling pixels
            sibling_idx = SHP_i[i][j], SHP_j[i][j]
            # Get the prior distribution for the pixel probability
            prior_dist = pm.Beta.dist(alpha=alpha, beta=beta)
            # Get the prior distribution for the sibling probabilities
            prior_sibling_dist = pm.Beta.dist(alpha=alpha, beta=beta)
            with pm.Model() as model:
                # Define the likelihood function for the pixel probability
                likelihood = pm.Bernoulli('likelihood', p=prior_dist, observed=prob_with_padding[i+1, j+1])
                # Define the likelihood function for the sibling probabilities
                sibling_likelihood = pm.Beta('sibling_likelihood', alpha=prior_sibling_dist.alpha + n_siblings[i, j] * alpha,
                                             beta=beta + n_siblings[i, j] * (1-alpha),
                                             observed=np.nanmean(prob_with_padding[sibling_idx]))
                # Compute posterior distribution for the pixel probability
                posterior_dist = pm.Beta('posterior', alpha=prior_dist.alpha + likelihood.sum(),
                                         beta=beta + likelihood.size - likelihood.sum())
                # Sample from the posterior distribution
                trace = pm.sample(1000, tune=1000)
                # Compute the mean of the posterior distribution
                reinforced_prob = pm.summary(trace)['mean']['posterior']
                # Store the reinforced probability in the output array
                reinforced_probs[i, j] = reinforced_prob
    return reinforced_probs


def reinforce_probabilities_bayes(prob_array, n_siblings, alpha, beta):
    # Create an empty array to hold the reinforced probabilities
    reinforced_probs = np.zeros_like(prob_array)
    
    # Loop over all the pixels in the input probability array
    for i in range(prob_array.shape[0]):
        for j in range(prob_array.shape[1]):
            # Get the indices of the sibling pixels
            sibling_idx = np.where(n_siblings[i, j, :] > 0)[0]
            
            # If there are no sibling pixels, use the original probability
            if len(sibling_idx) == 0:
                reinforced_probs[i, j] = prob_array[i, j]
                continue
            
            # Define the prior distribution for the pixel probability
            prior_dist = pm.Beta('prior', alpha=alpha, beta=beta)
            
            # Define the likelihood function for the pixel probability
            likelihood = pm.Bernoulli('likelihood', p=prior_dist, observed=prob_array[i, j])
            
            # Define the likelihood function for the sibling probabilities
            sibling_likelihood = pm.Beta('sibling_likelihood', alpha=alpha, beta=beta,
                                         observed=np.mean(prob_array[sibling_idx]))
            
            # Compute posterior distribution for the pixel probability
            posterior_dist = pm.Beta('posterior', alpha=prior_dist.alpha + likelihood.sum(),
                                     beta=prior_dist.beta + likelihood.size - likelihood.sum())
            
            # Sample from the posterior distribution
            trace = pm.sample(1000, tune=1000)
            
            # Get the posterior mean as the reinforced probability
            reinforced_prob = np.mean(trace['posterior'])
            
            # Store the reinforced probability in the output array
            reinforced_probs[i, j] = reinforced_prob
            
    return reinforced_probs

