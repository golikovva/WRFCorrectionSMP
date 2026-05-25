import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ClusterMapper(nn.Module):
    def __init__(self, mapping_file=None, target_coords=None, input_coords=None, 
                 weighted=False, save_mapping=False, save_name='meaner_mapping.npy', 
                 device='cpu', distance_metric='euclidean'):
        super().__init__()
        self.mapping = None
        self._target_slice = None
        self.distances = None
        self.reverse_distance = None
        self.denominator = None
        self.counts = None
        self.mask = None
        self.weighted = weighted
        self.device = device
        self.distance_metric = distance_metric

        if target_coords is not None and input_coords is not None:
            self.target_coords = target_coords.copy(order='C')
            self.input_coords = input_coords.copy(order='C')

        if mapping_file is not None:
            self.set_mapping_by_file(mapping_file)
        else:
            assert target_coords is not None and input_coords is not None
            self.create_mapping()
            if save_mapping:
                self.save_mapping(save_name)
        self._precompute_counts_mask()

        if self.weighted:
            self.calc_weights()
    @property
    def target_slice(self):
        if self._target_slice is None:
            self._target_slice = self.mapping.unique().long().cpu()
        return self._target_slice

    def set_mapping_by_file(self, mapping_file):
        data = np.load(mapping_file, allow_pickle=True).item()
        self.mapping = torch.from_numpy(data['indices']).long()
        if 'distances' in data:
            self.distances = torch.from_numpy(data['distances'])

    def create_mapping(self):
        """Unified method for creating mapping using NearestNeighbors"""
        if self.distance_metric == 'haversine':
            target_coords = np.radians(self.target_coords)
            input_coords = np.radians(self.input_coords)
        else:
            target_coords = self.target_coords
            input_coords = self.input_coords

        nearn = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=self.distance_metric)
        nearn.fit(target_coords)
        distances, indices = nearn.kneighbors(input_coords)
        
        self.mapping = torch.from_numpy(indices.squeeze()).long()
        self.distances = torch.from_numpy(distances.squeeze())

    def _precompute_counts_mask(self):
        self.counts = torch.bincount(self.mapping, minlength=self.target_coords.shape[0])
        self.mask = self.counts > 0

    def calc_weights(self):
        """Use precomputed distances from NearestNeighbors"""
        if self.distances is None:
            raise RuntimeError("Distances not calculated. Call create_mapping first.")

        self.reverse_distance = (1 / self.distances).to(self.device)
        self.denominator = torch.zeros(self.target_coords.shape[0], 
                                     device=self.device,
                                     dtype=self.reverse_distance.dtype)
        self.denominator.scatter_add_(0, self.mapping.to(self.device), self.reverse_distance)
        self.denominator = self.denominator.clamp(min=1e-6)

    def save_mapping(self, filename):
        data = {
            'indices': self.mapping.numpy(),
            'distances': self.distances.numpy() if self.distances is not None else None
        }
        np.save(filename, data)

    def forward(self, output, masked=True):
        if self.weighted:
            res =  self._forward_weighted(output)
        else:
            res = self._forward_mean(output)
        if masked:
            res = res[..., self.mask.to(output.device)]
        return res

    def _forward_mean(self, output):
        output_flat = output.flatten(-2, -1)
        mapping = self.mapping.expand_as(output_flat).to(output.device)
        summed = torch.zeros(*output_flat.shape[:-1], self.target_coords.shape[0],
                           device=output.device)
        summed.scatter_add_(-1, mapping, output_flat)
        return (summed / self.counts.to(output.device).clamp(min=1e-6))

    def _forward_weighted(self, output):
        output_flat = output.flatten(-2, -1)
        reverse_distance = self.reverse_distance.view(*([1]*(output_flat.dim()-1)), -1)
        weighted_output = output_flat * reverse_distance.to(output.device, dtype=output.dtype)
        mapping = self.mapping.expand_as(weighted_output).to(output.device)
        summed = torch.zeros(*weighted_output.shape[:-1], self.target_coords.shape[0],
                           device=output.device, dtype=output.dtype)
        summed.scatter_add_(-1, mapping, weighted_output)
        return (summed / self.denominator.to(output.device, dtype=output.dtype).clamp(min=1e-6))

    def to(self, device):
        self.device = device
        for attr in ['reverse_distance', 'denominator', 'mapping', 'counts', 'mask', 'distances']:
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))
        return self
    