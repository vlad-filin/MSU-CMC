import numpy as np
from numpy.linalg import slogdet, det, solve
from scipy.stats import multivariate_normal
from scipy.stats import norm
class MixtureModel:
    def __init__(self, n_components, diag=False):
        """
        Parametrs:
        ---------------
        n_components: int
        The number of components in mixture model

        diag: bool
            If diag is True, covariance matrix is diagonal
        """
        self.n_components = n_components  
        # bonus part
        self.diag = diag
        
    def _E_step(self, data):
        """
        E-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point
        """
        eps = 1e-15 
        
            # set self.q_z
        if self.diag:
            # bonus part
            N, n_components = self.q_z.shape
            #self.q_z  = np.ones(self.q_z.shape)
            w = self.w # n_comp
            mean = self.Mean # (n_comp, n features)
            _, D = mean.shape
            cov = self.Sigma # (n_comp, n_feat, n_feat)
            for c in range(n_components):
                rv = multivariate_normal(mean=mean[c,:], cov=np.diag(cov[c,...]), allow_singular=True)
                self.q_z[:,c] = rv.pdf(data)
            self.prob = self.q_z.copy()
            self.q_z = w[np.newaxis,:] * self.q_z + eps / n_components
            norma = self.q_z.sum(axis=1)
            self.q_z = self.q_z / norma[:,np.newaxis]
        else:
            #
            # 1 small broadcasting here + 1 small for
            #
            N, n_components = self.q_z.shape
            w = self.w # n_comp
            mean = self.Mean # (n_comp, n features)
            cov = self.Sigma # (n_comp, n_feat, n_feat)
            for c in range(n_components):
                rv = multivariate_normal(mean=mean[c,:], cov=cov[c,...], allow_singular=True)
                self.q_z[:,c] = rv.pdf(data)
            self.prob = self.q_z.copy() # for log likelihood
            self.q_z = w[np.newaxis,:] * self.q_z + eps / n_components
            norma = self.q_z.sum(axis=1)
            self.q_z = self.q_z / norma[:,np.newaxis]
            #print(' After norm\n', self.q_z)
                                         
            
                
    def _M_step(self, data):
        """
        M-step of the algorithm
        
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
            Array of data points. Each row corresponds to a single data point.
        """
        N, d = data.shape
        #print("k")
        # set self.w, self.Mean, self.Sigma
        n_comp, _ = self.Mean.shape
        self.w = self.q_z.mean(axis=0)
        norma = self.q_z.sum(axis=0)
        self.Mean = np.dot(self.q_z.T, data)/ norma[:,np.newaxis]
        if self.diag:
            # bonus part
            for k in range(n_comp):
                centered = (data - self.Mean[k, np.newaxis]) ** 2
                
                self.Sigma[k,...] = (np.diag((self.q_z[...,k][:,np.newaxis] * centered).sum(axis=0)) / 
                                     (self.q_z[..., k]).sum())
  

        else:
            for k in range(n_comp):
                
                centered = data - self.Mean[k, np.newaxis]
                #print("centered\n",centered)
                self.Sigma[k,...] = (np.dot(centered.T,self.q_z[...,k][:,np.newaxis] * centered)/
                                    (self.q_z[..., k]).sum())
          #  0/0
  
          #  print("Mean\n ", self.Mean)
           # print("Sigma\n ",self.Sigma)
            
            
    
    def EM_fit(self, data, max_iter=10, tol=1e-3,
               w_init=None, m_init=None, s_init=None, trace=False, seed=32, reg=False, min_sigma=5):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        w_init: numpy array shape(n_components)
        Array of the each mixture component initial weight

        Mean_init: numpy array shape(n_components, n_features)
        Array of the each mixture component initial mean

        Sigma_init: numpy array shape(n_components, n_features, n_features)
        Array of the each mixture component initial covariance matrix
        
        trace: bool
        If True then return list of likelihoods
        """
        # parametrs initialization
        np.random.seed(seed)
        n_components = self.n_components
        N, d = data.shape
        self.q_z = np.zeros((N, self.n_components))
        self.tol = tol
        
        # other initialization
        if w_init is None:
            self.w = np.ones(n_components) / n_components
        else:
            self.w = w_init

        if m_init is None:
            self.Mean = np.zeros((n_components, d))
            for k in range(n_components):
                self.ind = np.random.randint(0,N,size=int(20))
                self.Mean[k,...] = data[self.ind,...].mean(axis=0)
        else:
            self.Mean = m_init

        if s_init is None:
            self.Sigma = np.zeros((n_components, d, d))
            for k in range(n_components):
                #ind = np.random.randint(0,N,size=20)
                #self.Sigma[k,...] = np.cov(data[ind,...].T)
                #self.Sigma[k,...] = np.eye(d,d)
                v = np.amax(data[self.ind,...], axis=0) - np.amin(data[self.ind,...], axis=0)
                if reg is True:
                    v[v<5] = 5
                self.Sigma[k,...] = np.diag(v)
        else:
            self.Sigma = s_init
        
        log_likelihood_list = []
        tmp = np.inf
        # algo    
        for i in range(max_iter):
           # print("EM fit ", i)
            #print(i, self.Sigma)
            self._E_step(data)
            # Compute loglikelihood
            log_likelihood_list.append(self.compute_log_likelihood(data))

            # Perform M-step
            self._M_step(data)
            if reg is True:
                for k in range(n_components):
                    diag = np.diag(self.Sigma[k,...])
                    diag = min_sigma - diag
                    diag[diag<0] = 0
                    self.Sigma[k,...] += np.diag(diag)                         
            if abs(tmp - log_likelihood_list[-1]) < self.tol:
                break
            tmp = log_likelihood_list[-1]
        
        # Perform E-step
        # Compute loglikelihood
        self._E_step(data)
        log_likelihood_list.append(self.compute_log_likelihood(data))
        if trace:
            
            return self.w, self.Mean, self.Sigma, log_likelihood_list
        else:
            return self.w, self.Mean, self.Sigma
    
    def EM_with_different_initials(self, data, n_starts, max_iter=10, tol=1e-3,seed=56, reg=False, min_sigma=5):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.

        n_starts: int
        The number of algorithm running with different initials

        max_iter: int
        Maximum number of EM iterations

        tol: int
        The convergence threshold

        Returns:
        --------
        Best values for w, Mean, Sigma parameters
        """
        best_w, best_Mean, best_Sigma, max_log_likelihood = None, None, None, -np.inf
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e8, size=n_starts)
        for i in range(n_starts):
            w, mean, sigma, likelihood = self.EM_fit(data, max_iter=max_iter, seed=seeds[i],tol=tol, trace=True, reg=reg, min_sigma=min_sigma)
           # print("New\n", likelihood[-1])
            if likelihood[-1] > max_log_likelihood:
                best_w, best_Mean, best_Sigma, max_log_likelihood  = w, mean, sigma, likelihood[-1]
                #print("New\n", likelihood[-1])
                #print("new_mean\n", best_Mean)
        
        self.w = best_w
        self.Mean = best_Mean
        self.Sigma = best_Sigma
        
        return self.w, self.Mean, self.Sigma
    
    def compute_log_likelihood(self, data):
        """
        Parametrs:
        ---------------
        data: numpy array shape (n_samples, n_features)
        Array of data points. Each row corresponds to a single data point.
        """
        eps = 1e-15
        w = self.w # n_comp
        p = self.prob # N n_comp
        weighted_sum = (p * w[np.newaxis, :] + eps).sum(axis=1)
        return np.log(weighted_sum).sum()
                        
        
        return 