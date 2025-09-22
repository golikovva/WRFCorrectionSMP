import torch


class StandardScaler:
    def __init__(self):
        self.channel_means = None
        self.channel_stddevs = None
        self.channels = None
        self.mean = None
        self.stddev = None
        self.channels_dim = None

    def channel_inverse_transform(self, tensor, channels_dim=None, means=None, stds=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        if means is None:
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs

        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        for i in range(min(len(self.channels), len(tensor))):
            tensor[self.channels[i]] = tensor[self.channels[i]] * stds[i]
            tensor[self.channels[i]] = tensor[self.channels[i]] + means[i]
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

    def channel_fit(self, tensor, channels=None, channels_dim=1):
        self.channels_dim = channels_dim
        self.channel_means = []
        self.channel_stddevs = []
        if not channels:
            self.channels = range(tensor.shape[channels_dim])
        else:
            self.channels = channels
        tensor = torch.split(tensor, 1, dim=self.channels_dim)
        for i, channel in enumerate(tensor):
            if i in self.channels:
                self.channel_means.append(torch.mean(channel))
                self.channel_stddevs.append(torch.std(channel))

    def apply_scaler_channel_params(self, means, stds, channels=None):
        self.channel_means = means
        self.channel_stddevs = stds
        if channels is None:
            self.channels = list(range(len(means)))

    def channel_transform(self, tensor, channels_dim=None, means=None, stds=None):
        if not channels_dim:
            channels_dim = self.channels_dim
        if means is None:
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs

        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        j = 0
        for i, channel in enumerate(tensor):
            if i in self.channels:
                channel -= means[j]
                channel /= stds[j]
                tensor[i] = channel
                j += 1
        tensor = torch.cat(tensor, dim=channels_dim)
        return tensor

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
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) - means) / stds  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))

    def inverse_transform(self, tensor, means=None, stds=None, dims=None):
        if means is None:
            means = self.channel_means
        if stds is None:
            stds = self.channel_stddevs
        permutation = self.create_permutation(tensor.ndim, dims)
        out = (tensor.permute(permutation) * stds) + means  # returns an input copy
        return out.permute(*torch.argsort(torch.tensor(permutation)))

    def fit(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.stddev = tensor.std(0, unbiased=False, keepdim=True)
        self.max = tensor.max()


class SeasonalStandardScaler(StandardScaler):
    def seasonal_channel_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.channel_means[month]
        stds = self.channel_stddevs[month]
        tensor = self.transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor

    def seasonal_channel_inverse_transform(self, tensor, month, batch_dim, channels_dim):
        means = self.channel_means[month]
        stds = self.channel_stddevs[month]
        tensor = self.channel_inverse_transform(tensor, means, stds, [batch_dim, channels_dim])
        return tensor


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.range_min = feature_range[0]
        self.range_max = feature_range[1]
        self.tensor_min = None
        self.tensor_max = None
        self.channel_mins = []
        self.channel_maxs = []

    def channel_fit_transform(self, tensor, channels_dim=1):
        self.channel_mins = []
        self.channel_maxs = []
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        new_tensor = []
        for channel in tensor:
            self.channel_mins.append(torch.min(channel))
            self.channel_maxs.append(torch.max(channel))
            channel = (channel - self.channel_mins[-1]) / (self.channel_maxs[-1] - self.channel_mins[-1])
            new_tensor.append(channel)
        new_tensor = torch.cat(new_tensor, dim=channels_dim)
        return new_tensor

    def channel_fit(self, tensor, channels_dim=1):
        self.channel_mins = []
        self.channel_maxs = []
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        for channel in tensor:
            self.channel_mins.append(torch.min(channel))
            self.channel_maxs.append(torch.max(channel))

    def channel_transform(self, tensor, channels_dim=1):
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        new_tensor = []
        for i, channel in enumerate(tensor):
            channel = (channel - self.channel_mins[i]) / (self.channel_maxs[i] - self.channel_mins[i])
            new_tensor.append(channel)
        new_tensor = torch.cat(new_tensor, dim=channels_dim)
        return new_tensor

    def channel_inverse_transform(self, tensor, channels_dim=1):
        tensor = list(torch.split(tensor, 1, dim=channels_dim))
        new_tensor = []
        for i, channel in enumerate(tensor):
            channel = channel * (self.channel_maxs[i] - self.channel_mins[i]) + self.channel_mins[i]
            new_tensor.append(channel)
        new_tensor = torch.cat(new_tensor, dim=channels_dim)
        return new_tensor

    def fit_transform(self, tensor):
        self.tensor_min = torch.min(tensor)
        self.tensor_max = torch.max(tensor)
        X_std = (tensor - self.tensor_min) / (self.tensor_max - self.tensor_min)
        X_scaled = X_std * (self.range_max - self.range_min) + self.range_min
        return X_scaled

    def inverse_transform(self, tensor):
        if self.channel_mins is not None:
            for i in range(len(tensor)):
                tensor[i] = tensor[i] * (self.channel_maxs[i] - self.channel_mins[i]) + self.channel_mins[i]
            return tensor
