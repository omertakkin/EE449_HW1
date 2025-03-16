import numpy as np
import os

# Assuming samples_7.npy is in /content/drive/MyDrive/EE449/HW1
file_path = os.path.join('_given\data\data', 'samples_7.npy')

# input shape: [batch size, input_channels, input_height, input_width]
input = np.load(file_path)
# input shape: [output_channels, input_channels, filter_height, filter width]
# Assuming kernel.npy is also in /content/drive/MyDrive/EE449/HW1
kernel_path = os.path.join('_given\data\data', 'kernel.npy')
kernel = np.load(kernel_path) # Load kernel.npy from the specified path

import sys
sys.path.append('_given')
from utils import part2Plots


def my_conv2d(input, kernel):
    """
    Performs a 2D convolution (forward propagation) with no padding and stride 1.

    Args:
        input (np.ndarray): Input data of shape (batch_size, in_channels, in_height, in_width).
        kernel (np.ndarray): Convolutional kernel of shape (out_channels, in_channels, k_height, k_width).

    Returns:
        np.ndarray: The result of the convolution, of shape (batch_size, out_channels, out_height, out_width),
                    where:
                        out_height = in_height - kernel_height + 1,
                        out_width  = in_width - kernel_width + 1.
    """
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, kernel_in_channels, kernel_height, kernel_width = kernel.shape

    # Check if the input channels and kernel channels match
    if in_channels != kernel_in_channels:
        print("error")
        raise ValueError("The number of input channels must match the kernel's input channels.")

    # Compute output dimensions
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1

    # Initialize the output tensor with zeros
    output = np.zeros((batch_size, out_channels, out_height, out_width))

    # Iterate over the batch, output channels, and spatial locations to apply the convolution filter
    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    # Extract the current patch from the input
                    patch = input[b, :, i:i+kernel_height, j:j+kernel_width]
                    # Perform elementwise multiplication and sum over the channel and kernel dimensions
                    output[b, oc, i, j] = np.sum(patch * kernel[oc, :, :, :])

    print("Convolution is done!")
    return output


out = my_conv2d(input, kernel)
part2Plots(out)