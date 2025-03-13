"""Gaussian Process model implementation for hybrid modeling.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Kernel


class SubspaceKernel(Kernel):

    def __init__(self, base_kernel, ids_to_apply):
        """
        This is used to apply a kernel only to a subset of all available feature columns.
        I.e. it allows to apply different kernels to different features.
        e.g. RBF kernel to 'concentration features', Embedding Kernel to categorical features.

        :param base_kernel: The kernel to apply
        :param ids_to_apply: (list/nparray of integers) the indices tto apply the kernel to
        """
        if not isinstance(base_kernel, Kernel):
            raise ValueError("base_kernel has to be a {} instance".format(Kernel))

        self.base_kernel = base_kernel
        self.ids_to_apply = ids_to_apply

    def _subindex(self, X):
        if X is None:
            return None

        new = X[:, self.ids_to_apply]

        return new

    def __repr__(self):
        sub_kernel_str = self.base_kernel.__repr__()

        return "SubspaceKernel({} on {})".format(sub_kernel_str, self.ids_to_apply)

    def __call__(self, X, Y=None, eval_gradient=False):
        nX = self._subindex(X)
        nY = self._subindex(Y)

        return self.base_kernel.__call__(X=nX, Y=nY, eval_gradient=eval_gradient)

    def diag(self, X):
        nX = self._subindex(X)
        return self.base_kernel.diag(nX)

    def is_stationary(self):
        return self.base_kernel.is_stationary

    @property
    def requires_vector_input(self):
        return True

    @property
    def n_dims(self):
        return len(self.ids_to_apply)

    @property
    def hyperparameters(self):
        return self.base_kernel.hyperparameters

    @property
    def theta(self):
        return self.base_kernel.theta

    @theta.setter
    def theta(self, theta):
        self.base_kernel.theta = theta

    @property
    def bounds(self):
        return self.base_kernel.bounds


def fit_gp_model(X, y):
    n_features = X.shape[-1]

    # Apply RBF kernel to the normal features
    raw_feature_kernel = RBF(length_scale=[1e-1] * n_features, length_scale_bounds=(1e-2, 1e2))

    # The SubspaceKernel ensures that the RBF kernel is only applied to the first 10 features
    feature_kernel = SubspaceKernel(
        raw_feature_kernel, ids_to_apply=np.arange(0, n_features)
    )

    # Noise kernal
    noise_kernel = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))

    # Combine the kernel and allow for Noise
    full_kernel = 1**2 * feature_kernel + noise_kernel
    gp_model = GaussianProcessRegressor(kernel=full_kernel, n_restarts_optimizer=3)

    # Fit GP model
    gp_model.fit(X, y)
    return gp_model




