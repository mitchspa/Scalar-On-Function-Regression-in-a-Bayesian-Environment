import numpy as np
import matplotlib.pyplot as plt

def proj(K,T,T0,f): 
    tau = np.where(T0==1)[0]
    K_tau=K[:, tau]
    K_tau_tau = K[tau[:, None], tau]
    beta_tau = f[tau] 
    proj = f-np.dot(K_tau,np.dot(np.linalg.inv(K_tau_tau),beta_tau))
    return proj

class data_generator:
    def __init__(self, T, beta, n, kernel, sigma, plot, T0=None):
        self.T = T
        self.g = T.shape[0]
        self.K = np.fromfunction(np.vectorize(lambda s, t: kernel(self.T[s], self.T[t])), (self.g, self.g), dtype=int)
        self.n = n
        self.beta = beta
        self.sigma = sigma
        self.T0 = T0
        self.plot = plot

    def i_o(self):
        x = np.random.multivariate_normal(np.zeros(self.g), self.K, self.n)
        if self.T0 is None: 
            y = np.trapz(x * self.beta, self.T, axis=1) + np.random.normal(0, self.sigma, size=self.n)
        else:
            beta0 = proj(self.K, self.T, self.T0, self.beta)
            y = np.trapz(x * beta0, self.T, axis=1) + np.random.normal(0, self.sigma, size=self.n)
            
        if self.plot: 
            fig, ax = plt.subplots(2)
            ax[0].plot(self.T, x.T)
            if self.T0 is not None:
                ax[0].fill_between(self.T0, -4, 4, alpha=0.2, label='T0')
                ax[0].legend()
            ax[0].set_title('Functional Covariates')
            ax[1].plot(range(self.n), y)
            ax[1].set_title('Scalar Responses')
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        return x, y