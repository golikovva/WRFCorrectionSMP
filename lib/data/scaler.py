import torch


class StandardScaler:
    def __init__(self):
        self.means = None
        self.stddevs = None

    def apply_scaler_channel_params(self, means, stds):
        self.means = means
        self.stddevs = stds

    def to(self, device):
        self.means = self.means.to(device)
        self.stddevs = self.stddevs.to(device)
        return self

    @staticmethod
    def create_permutation(tensor_ndim, dims=None):
        permutation = list(range(tensor_ndim))
        if not hasattr(dims, '__iter__'):
            dims = [dims]
        dims_to_normalize = []
        for dim in dims:
            if dim is not None:
                permutation.remove(dim)
                dims_to_normalize.append(dim)
        permutation.extend(dims_to_normalize)
        return permutation

    def transform(self, tensor, means=None, stds=None, dims=None):
        if means is None:
            means = self.means
        if stds is None:
            stds = self.stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) - means) / stds  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))

    def inverse_transform(self, tensor, means=None, stds=None, dims=None):
        if means is None:
            means = self.means
        if stds is None:
            stds = self.stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) * stds) + means  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))


class SeasonalStandardScaler(StandardScaler):
    def seasonal_channel_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.means[month]
        stds = self.stddevs[month]
        tensor = self.transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor

    def seasonal_channel_inverse_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.means[month]
        stds = self.stddevs[month]
        tensor = self.inverse_transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor
