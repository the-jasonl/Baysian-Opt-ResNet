import numpy as np

from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel

class GaussianProcessOptimizer():
    """Bayesian function optimizer which uses a Gaussian process
    to model the expensive function, and expected improvement as
    the acquisition function.
    """

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
        self.bounds = np.array([[lb, ub]])
        self.noise = 0.1
        self.n_restarts = 10
        # Gaussian process to use for estimating the function
        self.gp = GaussianProcessRegressor(
            kernel=RationalQuadratic() + WhiteKernel(), 
            alpha=self.noise,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
        )
        
        self.X_samples = []
        self.Y_samples = []
        self.opt_x = None
        self.opt_y = np.inf
        self.X = np.arange(lb, ub, 0.01).reshape(-1, 1)

    def add_point(self, x, y):
        """Add a point to the history of sampled points."""
        self.X_samples.append(x)
        self.Y_samples.append(y)
        if y < self.opt_y:
            self.opt_y = y
            self.opt_x = x

    def best_point(self):
        """Returns best sampled point"""
        return self.opt_x

    def expected_improvement(self, X, X_sample, gpr):
        """Calculate expected improvement in X based on previous samples"""
        mu, sigma = gpr.predict(X, return_std=True)
        mu_sample = gpr.predict(np.array(X_sample).reshape(-1,1))          
        mu_sample_opt = np.min(mu_sample)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        imp = mu - mu_sample_opt
        flip = -1 # for minimization
        Z = flip * imp / sigma
        ei = flip * imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        return ei
    
    def next_point(self):
        """Get the point with the highest expected improvement."""
        min_val = np.inf
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -self.expected_improvement(X.reshape(-1,1), self.X_samples, self.gp)
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.lb, self.ub,size=self.n_restarts):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')        
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        return min_x.item()

    def fit(self):
        print("Fitting")
        return self.gp.fit(np.array(self.X_samples).reshape(-1,1), np.array(self.Y_samples).reshape(-1,1))