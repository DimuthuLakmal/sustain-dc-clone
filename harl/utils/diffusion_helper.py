import numpy as np

class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu=0.0, sigma=0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape.

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape).astype(np.float32)
        return noise