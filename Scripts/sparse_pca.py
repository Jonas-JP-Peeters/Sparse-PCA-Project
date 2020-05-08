import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import null_space
from IPython.display import clear_output
from sklearn.linear_model import LassoLars
import random
import warnings
warnings.filterwarnings('ignore')

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = np.abs(array.flatten())
#     if np.amin(array) < 0:
#         # Values cannot be negative:
#         array -= np.amin(array)
#     # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def reg_PCA(X, k = "all"): 
    """
    function takes an n x p feature matrix
    returns two arrays:
    - array with percentage of explained variance in first k principal directions (k_comp x 1)
    - array with principal directions (k_comp x p)
    """
    X = StandardScaler().fit_transform(X)
    if k == "all": k = min(X.shape[0],X.shape[1])
    pca = PCA(n_components = k)
    pca.fit(X)
    PEVs = pca.explained_variance_ratio_
    prin_comp = pca.components_
    EVs = pca.explained_variance_
    
    return PEVs, prin_comp, EVs

def threshold_PCA(X, thresh = 1e-1, k = "all"):
    """
    function takes
    - X: n x p feature matrix
    - thresh: float representing this non-zero cutoff
    - k: integer for number of principal directions wanted
    returns one array:
    - array with principal components in its columns (k x p)
    """
    if k == "all": k = min(X.shape[0],X.shape[1])

    pcs = reg_PCA(X)[1]
    pcs = (np.abs(pcs) >= thresh).astype(int) * pcs
    
    return pcs[:k]

def nonZeroLoad_PCA(X,j, k = "max"):
    """
    function takes
    - X: n x p feature matrix
    - j: integer for number of non-zero loadings,
    - k: integer for number of principal directions wanted
    returns three arrays:
    - array with percentages of explained variance in first k principal directions (k x 1)
    - array with principal directions (k x p)
    - array with explained variances
    """   
    X_scaled = StandardScaler().fit_transform(X)
    if k == "all": k = min(X.shape[0],X.shape[1])
    
    PCA_PEV, PCA_PC, PCA_EV = reg_PCA(X,min(X.shape[0],X.shape[1]))
    
    total_var = sum(PCA_EV)
    
    thresh_PCA_PC = np.empty((0,PCA_PC.shape[1]))
    thresh_PCA_PEV = []
    thresh_PCA_EV = []
    
    PCA_PC_sorted = np.sort(np.absolute(PCA_PC), axis = 1)
    for m in range(k):
        thresh = PCA_PC_sorted[m][-j]
        thresh_PC = (np.absolute(PCA_PC[m]) >= thresh).astype(int)*PCA_PC[m]
        thresh_PCA_PC = np.vstack((thresh_PCA_PC, thresh_PC))
    
    return thresh_PCA_PC

def SimpTrans(r, s, k, X_scaled):
    """
    function takes
    - Two directions r and s to transform
    - An integer k that restricts the possible directions
    - The scaled data matrix X
    returns two vectors and three floats
    - Two new directions r and s
    - Two floats representing the variances explained by the new directions r and s
    - One float representing the covariance of the new directions r and s
    """
    # Calculate the covariance matrix for the two directions
    cov = np.vstack((r,s)) @ np.cov(X_scaled.T) @ np.vstack((r,s)).T
    v_old_r, v_old_s, v_old_rs = cov[0,0], cov[1,1], cov[0,1]
    
    # Get a list of all possible betas
    poss_beta1 = [i/2**k for i in range(-2**k, 2**k+1, 1)]
    poss_beta2 = [2**k/i for i in range(-2**k, 2**k+1, 1) if i != 0]
    poss_beta = np.sort(list(set(poss_beta1 + poss_beta2)))
    
    # Calculate the norms of the two directions
    l2_r, l2_s = np.linalg.norm(r)**2, np.linalg.norm(s)**2
    
    # Find the beta that maximizes the variance in the normalized
    # direction of the first new principal component
    v_r, v_s, v_rs = cov[0,0], cov[1,1], cov[0,1]
    a = l2_s*v_rs
    b = np.sqrt(l2_r*l2_s)*(v_r - v_s)
    c = l2_r*v_rs
    discr = b**2 - 4*a*c
    beta_star = (-b+np.sqrt(discr))/(2*a)
    
    # Select beta from possible values that's closest to the optimal value
    beta_star = min(list(poss_beta), 
                    key = lambda x:abs(x-beta_star))
    
    # Calculate the two new directions
    if abs(beta_star) <= 1:
        r_new = 2**k*r + 2**k*beta_star*s
        s_new = 2**k*beta_star*l2_s*r - 2**k*l2_r*s
    else:
        r_new = 2**k*r/beta_star + 2**k*s
        s_new = 2**k*l2_s*r - 2**k*l2_r*s/beta_star
    
    # Calculate variances and covariance of the new directions
    P = np.array([[1, l2_s*beta_star],[beta_star, -l2_r]])
    cov_new = P.T @ cov @ P
    v_r, v_s, v_rs = cov_new[0,0], cov_new[1,1], cov_new[0,1]
    
    return r_new, s_new, v_r, v_s, v_rs

def pcPairFinder(cov, cache):
    """
    function takes
    - A covariance matrix
    - A cache containing the indices of excluded pcs
    returns two integers:
    - i_r: the index of the first principal component
    - i_s: the index of the second principal component
    """
    # Turn covariance matrix into a sparse lower triangular matrix
    grid = np.tril(cov, k = -1)
    vrs = np.diag(cov)
    # Flatten the grid and sort from largest cov to smallest cov
    srt = np.sort(np.ravel(grid), kind = 'heapsort')
    lst = list(np.flip(np.trim_zeros(srt)))
    mask = False
    
    # Find the pair of pcs that are not already in the cache and hav
    # the highest covariance in the grid
    while not np.any(mask):
        covar = lst.pop(0)
        index = np.argwhere(grid == covar)
        mask = np.ravel(np.invert(np.any(np.isin(index, cache), axis = 1, keepdims = True)))

    # Unpack the indices make sure that the variance of r is larger than the variance of s
    index = index[mask,:][0]
    i_r = index[np.argmax([vrs[i] for i in index])]
    i_s = index[np.argmin([vrs[i] for i in index])]
    return i_r, i_s

def simple_PCA(X, k, iters = 1):
    """
    function takes
    - An unscaled data matrix X
    - An integer k that restricts the possible directions
    - An integer indicating the number of iterations
    returns one array:
    - array with principal components in its columns (k x p)
    """
    # Retrieve the principal components and their covariance matrix
    n_obs, n_feats = X.shape
    q = np.identity(n_obs)
    for _ in range(n_feats//n_obs):
        q = np.hstack((q,np.identity(n_obs)))
    pcs = q[:n_obs,:n_feats]
    
    cov = np.cov(X.T)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Setup up globals for loop
    cov_pc = pcs @ cov @ pcs.T
    cache = []
    
    for _ in range(iters):
        while len(cache) <= len(pcs) - 2:
            clear_output(wait = True)
            print("Working on principal component ", len(cache) + 2,"/",len(pcs))
            
            # Find set of principal components to transform
            i_r, i_s = pcPairFinder(cov_pc, cache)
            r, s = pcs[i_r], pcs[i_s]
            cache += [i_r, i_s]
            
            # Transform the pair of components
            print("Determining new directions...")
            r_new, s_new, v_r, v_s, v_rs = SimpTrans(r, s, k, X_scaled)

            # Update the principal components
            pcs[i_r] = r_new
            pcs[i_s] = s_new
            
            # Update the grid matrix
            cov_pc[i_s, i_r], cov_pc[i_r, i_s] = v_rs, v_rs
    
    return pcs

def scotlass_pen(x, gamma):
    '''
    regularization penalty function for SCoTLASS
    
    function takes 
    - x: float
    - gamma: float
    
    function returns 
    - penalty value: float
    '''
    return (0.5 * x) * (1 + np.tanh(gamma*x))

def scotlass_obj(sigma, v, reg_param, gamma):
    '''
    objective function for SCoTLASS
    
    function takes
    - sigma: p x p covariance matrix
    - v: p x 1 vector
    - reg_param: regularization parameter (positive float)
    - gamma: float
    
    function returns
    - objective function value at v (p x 1 vector)
    '''
    varimax = (0.5*v.T) @ sigma @ v
    argpen = v.T @ np.tanh(1000*v)-reg_param
    penalty = gamma * scotlass_pen(argpen, gamma)
    return np.array(varimax - penalty).flatten()

def scotlass_grad(sigma, v, reg_param, gamma):
    '''
    gradient of objective function for SCoTLASS
    
    function takes
    - sigma: p x p covariance matrix
    - v: p x 1 vector
    - reg_param: regularization parameter (positive float)
    - gamma: float
    
    function returns
    - gradient of objective function at v
    '''
    # Setup of parameters
    mu = 1000
    
    # Calculate floats
    y = (v.T @ np.tanh(gamma*v))-reg_param
    q = 1 + np.tanh(gamma*y) + gamma*np.cosh(gamma*y)**(-2)*y
    
    # Calculate vectors
    z = np.tanh(gamma*v) + gamma * (np.diag(np.ravel(np.cosh(gamma*v)**(-2))) @ v)
    
    return (sigma @ v) - 0.5*mu*q*z 

def scotlassGradAsc(sigma, V, reg_param , x0 = 'default', 
                    alpha = 10**-3, max_iter = 20000, crit = 1e-1000):
    '''
    function takes
    - sigma: p x p covariance matrix
    - reg_param: regularization parameter (positive float); Default = sqrt(p)
    - x0: Initial value of the vector
    - alpha: Step size of gradient descent (<1 for convergence)
    - max_iters: max number of steps (positive integer)
    - crit: critical stopping value for the gradient descent algorithm
    
    function returns
    - v: first sparse principal direction (array of length p)
    - variance of data along v
    '''
    # Configure the parameters
    iters, delta, gamma = 1, 1, 1000
    num_obs, num_feat = sigma.shape
    
    # Configure the regularization parameter
#     reg_error = '''
#     The regularization parameter needs to be smaller than 
#     the square root of the number of features in the dataset.
#     '''
#     if reg_param == 'default': reg_param = np.sqrt(num_feat)
#     else: assert reg_param <= np.sqrt(num_feat), reg_error
    
    # Initialize the algorithm
    if x0 == 'default': x0 = np.ones(shape = (num_feat,1))
    elif x0 == 'random': x0 = np.random.rand(num_feat,1)
    else: pass
    v = x0/np.linalg.norm(x0)
    
    # Projected gradient descent
    # Stopping criteria:
    # (1) Maximum iterations reached
    # (2) Change in objective function negligible
    while iters < max_iter and delta > crit:
        
        # Update the vector
        v_new = v + alpha*scotlass_grad(sigma, v, reg_param, gamma)
        
        # Project loading vector back onto feasible set 
        # (vectors of l2 norm of 1 that are orthogonal to all other pcs)
        v_proj = V @ v_new
        v_proj = v_proj/(np.linalg.norm(v_proj))
        
        # Use the projected loading vector to retrieve the value of the
        # objective function
        old_obj = scotlass_obj(sigma, v, reg_param, gamma)[0]
        updated_obj = scotlass_obj(sigma, v_proj, reg_param, gamma)[0]

        # Calculate the difference in value of the objective function
        delta = [updated_obj - old_obj][0]
        #print(delta)
        # Update vector and number of iterations
        v, iters = v_proj, iters + 1
    
    # return loadings array v
    return v

def SCoTLASS(X, reg_param, x0 = 'default', alpha = 10**-3,
             max_iter = 200000, crit = 1e-1000):
    '''
    function takes
    - X: n x p dataset
    - reg_param: regularization parameter (positive float); Default = sqrt(p)
    - x0: Initial value of the vector
    - alpha: Step size of gradient descent (<1 for convergence)
    - max_iters: max number of steps (positive integer)
    - crit: critical stopping value for the gradient descent algorithm
    
    function returns
    - array with principal components in its columns (n x p)
    '''
    # Standardize the dataset
    X = StandardScaler().fit_transform(X)
    sigma = np.cov(X.T)
    num_feat = X.shape[1]
    num_samp = X.shape[0]
    
    # Initialize an array for principal components
    pcs = np.zeros(shape = (min(X.shape[0],X.shape[1]),X.shape[1]))
    
    # Initialize projection vector V
    V = np.identity(num_feat)
    
    for _ in range(len(pcs)):
        # Find the best principal component
        pc = scotlassGradAsc(sigma, V, reg_param, x0, alpha, max_iter, crit)
        pcs[_] = pc.T
        
        # Project the covariance matrix and x0 
        # on the orthogonal complement of previous pcs
        V = null_space(pcs[:_+1]) @ null_space(pcs[:_+1]).T
        sigma = V @ sigma
        x0 = V @ np.ones(shape = (num_feat,1))
    return pcs

def varimax(x):
    '''
    varimax penalty function
    
    function takes 1 x p vector
    
    function returns float
    '''
    #     p, m = B[None,:].shape
    #     P = m*np.sum(np.power(B, 4), axis = 0)
    #     Q = np.power(np.sum(np.power(B, 2), axis = 0), 2)
    #     return 1/p**2 * np.sum(P - Q)
    
    p = len(x)
    varimax = np.sum(np.power(x, 4))
    varimax -= (x.T @ x)[0]**2
    varimax = varimax/(p**2)
    
    return varimax

def varimax_grad(x):
    '''
    varimax penalty function gradient
    
    function takes 1 x p vector
    
    function returns 1 x p varimax gradient vector'''
    p = len(x)
    grad = 4*(np.power(x, 3) - (x.T @ x)* x)/(p**2)
    
    return grad

def scot_obj(sigma, v, reg_param):
    '''
    SCoT objective function
    
    function takes
    - sigma: p x p covariance matrix
    - v: 1 x p vector
    - reg_param: varimax regularization parameter (positive float)
    
    function returns float
    '''
    obj = v.T @ sigma @ v + (reg_param*varimax(v))[0]

    return obj

def scot_obj_grad(sigma, v, reg_param):
    '''
    SCoT objective function gradient
    
    function takes
    - sigma: p x p covariance matrix
    - v: 1 x p vector
    - reg_param: varimax regularization parameter (positive float)
    
    function returns 1 x p gradient vector
    '''
    
    p = len(v)
    grad = np.array((sigma @ v)) + reg_param*varimax_grad(v)
    return grad

def scotGradAsc(sigma, V, reg_param, x0 = 'default', 
                    alpha = 10**-3, max_iter = 20000, crit = 1e-1000):
    '''
    function takes
    - sigma: p x p covariance matrix
    - reg_param: regularization parameter (positive float); Default = sqrt(p)
    - x0: Initial value of the vector
    - alpha: Step size of gradient descent (<1 for convergence)
    - max_iters: max number of steps (positive integer)
    - crit: critical stopping value for the gradient descent algorithm
    
    function returns
    - v: first sparse principal direction (array of length p)
    - variance of data along v
    '''
    # Configure the parameters
    iters, delta, gamma = 1, 1, 1000
    num_obs, num_feat = sigma.shape
    
    # Initialize the algorithm
    if x0 == 'default': x0 = np.ones(shape = (num_feat,1))
    elif x0 == 'random': x0 = np.random.rand(num_feat,1)
    else: pass
    v = x0/np.linalg.norm(x0)
    
    # Projected gradient descent
    # Stopping criteria:
    # (1) Maximum iterations reached
    # (2) Change in objective function negligible
    while iters < max_iter and delta > crit:
        
        # Update the vector
        v_new = v + alpha*scot_obj_grad(sigma, v, reg_param)
        
        # Project loading vector back onto feasible set 
        # (vectors of l2 norm of 1 that are orthogonal to all other pcs)
        v_proj = V @ v_new
        v_proj = v_proj/(np.linalg.norm(v_proj))
        
        # Use the projected loading vector to retrieve the value of the
        # objective function
        old_obj = scot_obj(sigma, v, reg_param)[0]
        updated_obj = scot_obj(sigma, v_proj, reg_param)[0]

        # Calculate the difference in value of the objective function
        delta = [updated_obj - old_obj][0]
        
        # Update vector and number of iterations
        v, iters = v_proj, iters + 1
    
    # return loadings array v
    return v

def SCoT(X, reg_param, x0 = 'default', alpha = 10**-3,
             max_iter = 200000
         
         , crit = 1e-1000):
    '''
    function takes
    - X: n x p dataset
    - reg_param: regularization parameter (positive float)
    - x0: Initial value of the vector
    - alpha: Step size of gradient descent (<1 for convergence)
    - max_iters: max number of steps (positive integer)
    - crit: critical stopping value for the gradient descent algorithm
    function returns
    
    - array with principal components in its columns (n x p)
    '''
    # Standardize the dataset
    X = StandardScaler().fit_transform(X)
    sigma = np.cov(X.T)
    num_feat = sigma.shape[0]
    
    # Initialize an array for principal components
    pcs = np.zeros(shape = (min(X.shape[0],X.shape[1]),X.shape[1]))
    
    # Initial vector for gradient descent
    if x0 == 'default': x0 = np.ones(shape = (num_feat,1))
    elif x0 == 'random': x0 = np.random.rand(num_feat,1)
    else: pass
    v = x0/np.linalg.norm(x0)
    
    V = np.identity(X.shape[1])
    
    for _ in range(len(pcs)):
        # Find the best principal component
        pc = scotGradAsc(sigma, V, reg_param, x0, alpha, max_iter, crit)
        pcs[_] = pc.T
        
        # Project the covariance matrix and x0 
        # on the orthogonal complement of previous pcs
        V = null_space(pcs[:_+1]) @ null_space(pcs[:_+1]).T
        sigma = V @ sigma
        x0 = V @ np.ones(shape = (num_feat,1))
        x0 = x0/np.linalg.norm(x0)
        
    return pcs

def SVDProblem(B, X):
    '''
    function takes
    - X: n x p dataset
    - B: p x n matrix
    
    function returns
    - A: p x n matrix
    '''
    U, D, V = np.linalg.svd(X.T @ X @ B)

    return U @ V.T

def SPCA(X, reg_param, reg_param1, max_iter = 1000, crit = 1e-10):
    '''
    function takes
    - X: n x p dataset
    - reg_param: regularization parameter (positive float)
    - x0: Initial value of the vector
    - alpha: Step size of gradient descent (<1 for convergence)
    - max_iters: max number of steps (positive integer)
    - crit: critical stopping value for the gradient descent algorithm
    
    function returns
    - array with principal components in its columns (n x p)
    '''
    # (1) Let A start at V[,1:k] the loadings of the first 
    # k ordinary principal components
    A = reg_PCA(X)[1].T
    B = np.zeros_like(A)
    
    
    # Normalize the dataset
    X = StandardScaler().fit_transform(X)
    
    iters, delta, A_old = 1, 1, A
    while iters < max_iter and delta > crit:
        # (2) Given a fixed A = [alpha_1, ..., alpha_k], solve the elastic
        # net problem for j = 1, 2, ..., k
        for i in range(len(A)):
            B[i] = LARS_EN(X @ A[:,i].reshape(-1,1), # Target variable
                           X,                        # Data
                           reg_param,                # l2-norm reg param
                           reg_param1)               # l1-norm reg param
        
        # (3) For a fixed B = [beta_1, ..., beta_k], compute the SVD of 
        # X^TXB = UDV^T then update A = UV^T
        A = SVDProblem(B.T, X)

        # (4) Repeat steps (2) and (3) until convergence
        #iters, delta, A_old = iters + 1, np.linalg.norm(A - A_old), A
        iters, delta, A_old = iters + 1, np.linalg.norm(A - A_old), A
        #print(f"{iters} delta: {delta}")
    
    # (5) Normalize the altered principal components
    
    return (A/np.linalg.norm(A, axis = 1)).T

def LARS_EN(Y, X, reg_param, reg_param1):
    '''
    function takes
    - Y: p x 1 target variable
    - X: n x p dataset
    - reg_param: regularization parameter for l2-norm
    - reg_param1: regularization parameter for l1-norm
    
    function returns
    - beta: 1 x p vector with coefficients
    '''
    # Find the number of features
    p = X.shape[1]
    
    # Create the artificial dataset for the naïve elastic net
    X = np.power(1 + reg_param, -0.5) * np.vstack((X, np.sqrt(reg_param)*np.identity(p)))
    Y = np.vstack((Y, np.zeros(shape = (p,1))))
    gamma = reg_param1/np.sqrt(1 + reg_param)
    
    # Center X
    X = StandardScaler(with_std = False).fit_transform(X)
    
    # Use the LARS (Efron 2004) algorithm to solve this lasso regression
    lasso = LassoLars(alpha = gamma,
                      fit_intercept = False,
                      max_iter = 1000)
    lasso.fit(X, Y)
    
    # Transform the found coefficients in the elastic net coefficients
    beta = lasso.coef_/np.sqrt(1 + reg_param)
    
    return beta

def Adj_Var(X, PCs):
    adj_EVs = []
    X_prior = np.zeros((X.shape[1],X.shape[1]))
    for i in range(0,len(PCs)):
        V = np.reshape(np.array([PCs[j] for j in range(i+1)]),(i+1,len(PCs[i])))
        X_k = X @ V.T @ np.linalg.inv(V @ V.T) @ V
        adj_EV_k = (np.trace(X_k.T@X_k) - np.trace(X_prior.T @ X_prior))/(X.shape[0]-1)
        adj_EVs.append(adj_EV_k)
        X_prior = X_k
    return np.array(adj_EVs)