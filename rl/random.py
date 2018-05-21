from __future__ import division
import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass

class AnnealedSinusoidExponential(RandomProcess):
    '''
    Idea from this paper: http://cs231n.stanford.edu/reports/2017/pdfs/616.pdf part 5.1
    Can't be ran alone. Meant to be inherited by another random process

    -  eps_max is initial value
    -  eps_min is final value
    -  eps_d is decay rate
    -  nb_valleys is number of valleys in function
    -  nb_steps_annealing is number of steps before annealing is complete
    '''
    def __init__(self, eps_max=1.0, eps_min=0.0, eps_d=0.998, nb_valleys=25, n_steps_annealing=5000):
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_d   = eps_d
        self.nb_valleys = nb_valleys
        self.n_steps_annealing = n_steps_annealing
        self.n_steps = 0

        self.eps0 = self.eps_max - self.eps_min

    @property
    def current_anneal_value(self):
        out = self.eps_min + self.eps0 * (self.eps_d ** self.n_steps) * 0.5 * (1. + 
                    np.cos(2 * np.pi * self.n_steps * self.nb_valleys / self.n_steps_annealing))
        return out

class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

class OUProcessWithRandomStart(OrnsteinUhlenbeckProcess):
    def reset_states(self):
        self.x_prev = np.random.normal(self.mu, self.current_sigma, size=self.size)


class GaussianSinusoidProcess(AnnealedSinusoidExponential):
    def __init__(self, mu=0., sigma=1., sigma_min=None, eps_d=0.998, nb_valleys=25, n_steps_annealing=1000, size=1):
        super(GaussianSinusoidProcess, self).__init__(eps_max=sigma, eps_min=sigma_min, eps_d=eps_d, nb_valleys=nb_valleys, n_steps_annealing=n_steps_annealing)

        self.mu = mu
        self.n_steps = 0
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_anneal_value, self.size)
        self.n_steps += 1
        return sample