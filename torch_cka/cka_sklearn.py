"""
    CKA-based calculations for sklearn MLPClassifier models.
"""

import torch as ch
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np
from tqdm import tqdm
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from .utils import add_colorbar


def collect_all_activations(model: MLPClassifier,
                            X: np.ndarray,
                            model_layers: List[int] = None) -> List[np.ndarray]:
    """
    Collects all activations from all layers of a model.
    :param model: (MLPClassifier) sklearn model
    :param X: (np.ndarray) X to run the model on
    :param model_layers: (List) List of layers to extract features from
    :return: (List[np.ndarray]) list of all activations
    """
    wanted_activations = {}

    # Initialize first layer
    activation = X

    # Forward propagate
    hidden_activation = ACTIVATIONS[model.activation]
    for i in range(model.n_layers_ - 1):
        activation = safe_sparse_dot(activation, model.coefs_[i])

        # Store activation if requested
        if model_layers is None or i in model_layers:
            wanted_activations[i] = activation
        
        activation += model.intercepts_[i]
        if i != model.n_layers_ - 2:
            hidden_activation(activation)
    output_activation = ACTIVATIONS[model.out_activation_]
    output_activation(activation)

    # Store activation if requested
    if model_layers is None or model.n_layers_ in model_layers:
        wanted_activations[model.n_layers_] = activation

    return wanted_activations


class CKA_sklearn:
    def __init__(self,
                 model1: MLPClassifier,
                 model2: MLPClassifier,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[int] = None,
                 model2_layers: List[int] = None):
        """

        :param model1: (MLPClassifier) Neural Network 1
        :param model2: (MLPClassifier) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(model1.n_layers_) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU will thank you :)")

        self.model1_layers = model1_layers

        if len(model2.n_layers_) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU will thank you :)")

        self.model2_layers = model2_layers

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = ch.ones(N, 1).cuda()
        result = ch.trace(K @ L).cuda()
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def compare(self,
                data1: np.ndarray,
                data2: np.ndarray = None) -> Dict:
        """
        Computes the feature similarity between the models on the
        given data.
        :param data1: (np.ndarray)
        :param data2: (np.ndarray) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if data2 is None:
            warn("Data for Model 2 is not given. Using the same data for both models.")
            data2 = data1

        self.model1_info['Dataset'] = data1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = data2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.n_layers_))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.n_layers_))

        hsic_matrix = ch.zeros(N, M, 3)

        # Function to make forward pass with data and collect intermediate activations
        self.model1_features = collect_all_activations(self.model1, data1, self.model1_layers)
        self.model2_features = collect_all_activations(self.model2, data2, self.model2_layers)
        
        for i, (_, feat1) in enumerate(self.model1_features.items()):
            feat1 = feat1
            X = feat1.flatten(1)
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            hsic_matrix[i, :, 0] += self._HSIC(K, K)

            for j, (_, feat2) in enumerate(self.model2_features.items()):
                feat2 = feat2
                Y = feat2.flatten(1)
                L = Y @ Y.t()
                L.fill_diagonal_(0)
                assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                hsic_matrix[i, j, 1] += self._HSIC(K, L)
                hsic_matrix[i, j, 2] += self._HSIC(L, L)

        hsic_matrix = hsic_matrix[:, :, 1] / (hsic_matrix[:, :, 0].sqrt() *
                                                        hsic_matrix[:, :, 2].sqrt())

        assert not np.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"

        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": hsic_matrix.detach().cpu().numpy(),
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],
        }
