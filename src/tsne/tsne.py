import numpy as np


def neg_sq_euclidean_dist(a, b):
    """
    Returns the negative squared Euclidean distance between two vectors a and b.
    """
    euclidean_dist = np.sqrt(((a - b) ** 2).sum(axis=-1))
    return -euclidean_dist ** 2

def pairwise_distance(X):
    """
    Returns the pairwise distance matrix for a matrix X as a collection of row vectors.
    """
    return neg_sq_euclidean_dist(X[None, :, :], X[:, None, :])

def prob_matrix(dist_X, sigmas):
    """
    Returns the matrix of conditional probabilities p_j|i.
    :param dist_X: the pairwise distance matrix
    :param sigmas: a vector of sigma values corresponding to each row of the distance matrix
    """
    p = np.exp(dist_X / (2*sigmas.reshape(-1, 1)))
    np.fill_diagonal(p, 0)
    return p / p.sum(axis=1).reshape(-1, 1)

def perplexity(prob_X):
    """
    Calculates the perplexity of each row of the probability matrix.
    :param prob_X: the conditional probability matrix
    :return: a vector of perplexity values
    """
    entropy = -np.sum(prob_X * np.log2(prob_X), axis=1)
    return 2**entropy

def binary_search(f, target_perplexity, lower=1e-10, upper=1000, tol=1e-8, max_iter=10000):
    """
    Performs a binary search for the value of sigma that corresponds to the target perplexity.
    :param f: function to calculate perplexity
    :param target_perplexity: the specified perplexity value
    :param lower: initial lower bound
    :param upper: initial upper bound
    :param tol: tolerance to determine if the perplexity value is close enough to the target
    :param max_iter: maximum number of iterations of the loop
    :return: the optimal value of sigma
    """
    for i in range(max_iter):
        sigma = (lower + upper) / 2
        perp = f(sigma)
        if abs(perp - target_perplexity) < tol:
            return sigma
        if perp > target_perplexity:
            upper = sigma
        else:
            lower = sigma
    return sigma

def get_sigmas(dist_X, target_perplexity):
    """
    Finds the sigma for each row of the distance matrix based on the target perplexity.
    :param dist_X: the pairwise distance matrix
    :param target_perplexity: the specified perplexity value
    :return: a vector of sigma values corresponding to each row of the distance matrix
    """
    nrows = dist_X.shape[0]
    sigmas = np.zeros(nrows)

    for i in range(nrows):
        f = lambda sigma: perplexity(prob_matrix(dist_X[i:i + 1, :], np.array(sigma)))
        best_sigma = binary_search(f, target_perplexity)
        sigmas[i] = best_sigma
    return sigmas

def get_pmatrix(X, perplexity):
    """
    Calculates the final probability matrix using the pairwise affinities/conditional probabilities.
    :param M: the matrix of data to be converted
    :param perplexity: the specified perplexity
    :return: the joint probability matrix p_ij
    """
    # get the pairwise distances
    dist = pairwise_distance(M)
    # get the sigmas
    sigmas = get_sigmas(dist, perplexity)
    # get the matrix of conditional probabilities
    prob = prob_matrix(dist, sigmas)
    p = (prob + prob.T) / (2*prob.shape[0])
    return p
