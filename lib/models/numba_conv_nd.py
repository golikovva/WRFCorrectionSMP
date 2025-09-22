import numpy as np
from numba import njit, prange


def normalize_kernel(kernel):
    """Normalize the convolution kernel to make its sum equal to one."""
    kernel_sum = np.nansum(kernel)
    if kernel_sum != 0:
        return kernel / kernel_sum
    else:
        return kernel


@njit(parallel=True)
def conv4d_numba(input_data, kernel):
    """
    Perform 4D convolution on input_data with the given kernel, ignoring NaN values.

    Args:
        input_data (np.ndarray): 4D input data array with possible NaN values.
        kernel (np.ndarray): 4D convolution kernel.

    Returns:
        np.ndarray: The result of the 4D convolution.
    """
    kernel_shape = kernel.shape
    input_shape = input_data.shape

    # Calculate padding sizes
    pad_width = [(k // 2, k // 2) for k in kernel_shape]

    # Initialize output array
    output_data = np.zeros(input_shape, dtype=np.float64)

    # Pad the input data with NaNs manually
    padded_input = np.full((input_shape[0] + 2 * pad_width[0][0],
                            input_shape[1] + 2 * pad_width[1][0],
                            input_shape[2] + 2 * pad_width[2][0],
                            input_shape[3] + 2 * pad_width[3][0]), np.nan)

    padded_input[pad_width[0][0]:pad_width[0][0] + input_shape[0],
    pad_width[1][0]:pad_width[1][0] + input_shape[1],
    pad_width[2][0]:pad_width[2][0] + input_shape[2],
    pad_width[3][0]:pad_width[3][0] + input_shape[3]] = input_data

    # Perform convolution
    for i in prange(input_shape[0]):
        for j in range(input_shape[1]):
            for k in range(input_shape[2]):
                for l in range(input_shape[3]):
                    # Extract the sub-region for the current position
                    sub_region = padded_input[i:i + kernel_shape[0],
                                 j:j + kernel_shape[1],
                                 k:k + kernel_shape[2],
                                 l:l + kernel_shape[3]]
                    # Apply the kernel only to the valid (non-NaN) values
                    weighted_sum = 0.0
                    total_weight = 0.0
                    for ki in range(kernel_shape[0]):
                        for kj in range(kernel_shape[1]):
                            for kk in range(kernel_shape[2]):
                                for kl in range(kernel_shape[3]):
                                    if not np.isnan(sub_region[ki, kj, kk, kl]):
                                        weight = kernel[ki, kj, kk, kl]
                                        weighted_sum += sub_region[ki, kj, kk, kl] * weight
                                        total_weight += weight
                    if total_weight > 0:
                        output_data[i, j, k, l] = weighted_sum / total_weight

    return output_data


@njit(parallel=True)
def conv3d_numba(input_data, kernel):
    """
    Perform 4D convolution on input_data with the given kernel, ignoring NaN values.

    Args:
        input_data (np.ndarray): 3D input data array with possible NaN values.
        kernel (np.ndarray): 3D convolution kernel.

    Returns:
        np.ndarray: The result of the 3D convolution.
    """
    kernel_shape = kernel.shape
    input_shape = input_data.shape

    # Calculate padding sizes
    pad_width = [(k // 2, k // 2) for k in kernel_shape]

    # Initialize output array
    output_data = np.zeros(input_shape, dtype=np.float64)

    # Pad the input data with NaNs manually
    padded_input = np.full((input_shape[0] + 2 * pad_width[0][0],
                            input_shape[1] + 2 * pad_width[1][0],
                            input_shape[2] + 2 * pad_width[2][0]), np.nan)

    padded_input[pad_width[0][0]:pad_width[0][0] + input_shape[0],
    pad_width[1][0]:pad_width[1][0] + input_shape[1],
    pad_width[2][0]:pad_width[2][0] + input_shape[2]] = input_data

    # Perform convolution
    for i in prange(input_shape[0]):
        for j in range(input_shape[1]):
            for k in range(input_shape[2]):
                # Extract the subregion for the current position
                sub_region = padded_input[i:i + kernel_shape[0],
                             j:j + kernel_shape[1],
                             k:k + kernel_shape[2]]
                # Apply the kernel only to the valid (non-NaN) values
                weighted_sum = 0.0
                total_weight = 0.0
                for ki in range(kernel_shape[0]):
                    for kj in range(kernel_shape[1]):
                        for kk in range(kernel_shape[2]):
                            if not np.isnan(sub_region[ki, kj, kk]):
                                weight = kernel[ki, kj, kk]
                                weighted_sum += sub_region[ki, kj, kk] * weight
                                total_weight += weight
                if total_weight > 0:
                    output_data[i, j, k] = weighted_sum / total_weight

    return output_data


if __name__ == '__main__':
    from time import time

    # Пример использования
    input_data = np.random.rand(366, 34, 5, 5)
    input_data[1, 1, 1, 1] = np.nan  # Вставляем NaN для тестирования
    kernel = np.ones((3, 3, 3, 3))  # Пример ядра свертки

    # Нормализация ядра
    kernel = normalize_kernel(kernel)

    # Выполнение свертки
    start = time()
    output = conv4d_numba(input_data, kernel)
    print(output)
    print(time() - start)
