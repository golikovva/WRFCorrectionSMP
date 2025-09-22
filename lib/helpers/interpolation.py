import torch
import numpy as np
from sklearn.neighbors import KDTree, BallTree


class InvDistTree(torch.nn.Module):
    def __init__(self, x, q, leaf_size=10, n_near=6, sigma_squared=None,
                 distance_metric='euclidean', inv_dist_mode='gaussian', device='cpu'):
        super().__init__()

        self.ix = None
        self.weights = None
        self.distances = None
        self.dist_mode = inv_dist_mode
        self.x = np.asarray(x)
        self.q = np.asarray(q)
        self.k = 1
        self.device = device
        self.leaf_size = leaf_size
        self.tree = self.build_tree(distance_metric)  # KDTree(x, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(n_near, sigma_squared)
        self.to(device)

    def build_tree(self, distance_metric):
        if distance_metric == 'euclidean':
            self.tree = KDTree(self.x, leaf_size=self.leaf_size)
        elif distance_metric == 'haversine':
            self.q = np.radians(self.q)
            self.tree = BallTree(np.radians(self.x), leaf_size=self.leaf_size, metric=distance_metric)
        else:
            raise NotImplementedError
        return self.tree

    def calc_interpolation_weights(self, n_near=6, sigma_squared=None):
        self.distances, self.ix = self.tree.query(self.q, k=n_near)
        if n_near == 1:
            self.distances = self.distances[:, None]
            self.ix = self.ix[:, None]
        if np.where(self.distances < 1e-10)[0].size != 0:
            print('Zeros in indices!')
        self.weights = self.calc_dist_coefs(self.distances, sigma_squared)
        self.weights = self.weights / torch.sum(self.weights, dim=-1, keepdim=True)
        self.weights = torch.nan_to_num(self.weights, 1/n_near)
        self.weights = self.weights.type(torch.float).to(self.device)

    def calc_dist_coefs(self, dist, sigma_squared=None):
        if self.dist_mode == 'inverse':
            return torch.from_numpy(1 / dist)
        elif self.dist_mode == 'gaussian':
            sigma_squared = sigma_squared if sigma_squared else np.square(np.median(self.distances)) / 9 / self.k
            return gauss_function(dist, sigma_squared=sigma_squared)
        elif self.dist_mode == 'LinearNN':  # todo
            raise NotImplementedError

    def __call__(self, z):
        if z.shape[-1]*z.shape[-2] == len(self.x):
            res = self(z.flatten(-2, -1)).view(*z.shape[:-2], *self.q.shape[:-1])
        else:
            res = (z[..., self.ix] * self.weights).sum(-1)
        return res

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        s = mask_shape
        assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
        mask = torch.ones([s[-1] * s[-2]])
        mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
        mask = mask.reshape(*s).to(self.device)
        return mask
    
def gauss_function(x, sigma_squared=1):
    if isinstance(x, np.ndarray):
        x_torch = torch.from_numpy(x)
    else:
        x_torch = x
    f_x = 1 / np.sqrt(2*np.pi*sigma_squared) * torch.exp(-0.5 * x_torch * x_torch / sigma_squared)
    return f_x

def interpolate_data_on_scat_by_time_torch(data, data_times, scatter_times, scatter_mask=None, device='cpu'):
    '''
    data.shape == ..., sl, c, h, w
    data_times.shape == ..., sl
    scatter_times.shape == ..., d, h, w
    scat_mask.shape == ..., d, h, w
    '''
    if scatter_mask is None:
        scatter_mask = torch.ones_like(scatter_times)
    # Convert time difference to hours (ensure float for division)
    t_scatter = (scatter_times - data_times[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).float() / 3600 * scatter_mask  # Convert seconds to hours

    # Compute time indices for interpolation
    i0 = torch.floor(t_scatter).long()
    i0 = torch.clamp(i0, 0, data_times.shape[-1] - 1)
    i1 = torch.clamp(i0 + 1, 0, data_times.shape[-1] - 1)

    # Compute interpolation weights
    frac = t_scatter - i0
    weight0 = 1 - frac
    weight1 = frac

    n = data.ndim
    sl, c, h, w = data.shape[-4:]
    trailing_dims = data.shape[:-4]

    # Permute dimensions for easier indexing
    wrf_t = data.permute(n-2, n-1, n-3, *range(n-4, -1, -1))

    # Construct index tensors
    h_range = torch.arange(h, device=device).unsqueeze(1)  # Shape: (h, 1)
    w_range = torch.arange(w, device=device)  # Shape: (w,)

    indices_i0 = [h_range, w_range, slice(None), i0]
    indices_i0.extend(torch.arange(i, device=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) for i in trailing_dims)
    indices_i0 = tuple(indices_i0)
    wrf_i0 = wrf_t[indices_i0]

    indices_i1 = [h_range, w_range, slice(None), i1]
    indices_i1.extend(torch.arange(i, device=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) for i in trailing_dims)
    indices_i1 = tuple(indices_i1)
    wrf_i1 = wrf_t[indices_i1]

    # Perform linear interpolation
    interpolated = wrf_i0 * weight0.unsqueeze(-1) + wrf_i1 * weight1.unsqueeze(-1)

    # Reorder back to original shape
    interpolated_wrf = interpolated.permute(*range(n - 3), n-1, n-3, n-2) 
    return interpolated_wrf * scatter_mask.unsqueeze(-3).expand_as(interpolated_wrf)


def create_mask_by_nearest_to_nans(wind, coords, fill_value=0):
    nn_ids = InvDistTree(coords, coords, n_near=9).ix
    s = wind.shape
    num_neighbour_nans = np.isnan(wind.reshape(*s[:-2], -1)[..., nn_ids]).sum(-1)
    mask = np.ones_like(wind.reshape(*s[:-2], -1))
    mask[np.where(num_neighbour_nans > 0)] = fill_value
    return mask.reshape(*s)

