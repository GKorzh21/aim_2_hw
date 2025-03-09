import numpy as np


class BaseSampler:
    def __init__(self, max_samples=1.0, bootstrap=False, random_state=None):
        """
        Parameters
        ----------
        bootstrap : Boolean
            if True then use bootstrap sampling
        max_samples : float in [0;1]
            proportion of sampled examples
        """
        self.random_state = np.random.RandomState(random_state)
        self.bootstrap = bootstrap
        self.max_samples = max_samples

    def sample_indices(self, n_objects):
        """
        Parameters
        ----------
        n_objects : int > 0
            number of sampling objects
        """
        indexes = self.random_state.choice(
            a=n_objects,
            size=int(n_objects*self.max_samples), 
            replace=self.bootstrap
        )

        return indexes

    def sample(self, x, y=None):
        # abstract method
        raise NotImplementedError


class ObjectSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled, y_sampled : numpy ndarrays of shape (n_samples, n_features) and (n_samples,)
        """
        
        indexes = self.sample_indices(n_objects=x.shape[0])

        x_samples = x[indexes]
        y_samples = y[indexes] if y is not None else None

        return  x_samples, y_samples



class FeatureSampler(BaseSampler):
    def __init__(self, max_samples=1.0, bootstrap=True, random_state=None):
        super().__init__(max_samples=max_samples, bootstrap=bootstrap, random_state=random_state)

    def sample(self, x, y=None):
        """
        Parameters
        ----------
        x : numpy ndarray of shape (n_objects, n_features)
        y : numpy ndarray of shape (n_objects,)

        Returns
        -------
        x_sampled : numpy ndarrays of shape (n_objects, n_features_sampled)
        """
        
        indexes = self.sample_indices(n_objects=x.shape[1])

        x_samples = x[:, indexes]

        return  x_samples
