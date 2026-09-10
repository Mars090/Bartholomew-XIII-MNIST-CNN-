import numpy as np #i luh numpy

def correlate2d(input_array, kernel, mode='valid'):
    """
    Perform a 2D cross-correlation between input_array and kernel.

    Parameters:
    - input_array: 2D numpy array representing the input matrix.
    - kernel: 2D numpy array representing the kernel/filter.
    - mode: 'valid' or 'full' determines output size and padding behavior.

    Returns:
    - output_array: 2D numpy array result of correlation.
    """
    # Get dimensions of input and kernel
    input_rows, input_cols = input_array.shape
    kernel_rows, kernel_cols = kernel.shape
    
    # Determine output size and pad input if needed based on mode
    if mode == 'full':
        # Output size is larger than input due to full padding
        output_rows = input_rows + kernel_rows - 1
        output_cols = input_cols + kernel_cols - 1
        
        # Pad input with zeros on all sides to allow full convolution
        pad_height = kernel_rows - 1
        pad_width = kernel_cols - 1
        padded_input = np.pad(input_array, 
                              ((pad_height, pad_height), (pad_width, pad_width)), 
                              mode='constant')
    elif mode == 'valid':
        # Output size is smaller or equal (no padding)
        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1
        padded_input = input_array
    else:
        # Unsupported mode given
        raise ValueError(f"Unsupported mode: {mode}")
    
    # Initialise output array with zeros
    output_array = np.zeros((output_rows, output_cols))
    
    # Slide kernel over the input array
    for i in range(output_rows):
        for j in range(output_cols):
            # Extract current region of the input that overlaps kernel
            region = padded_input[i:i + kernel_rows, j:j + kernel_cols]
            # Compute element-wise product and sum to get correlation value
            output_array[i, j] = np.sum(region * kernel)
    
    return output_array

def convolve2d(input_array, kernel, mode='valid'):
    """
    Perform a 2D convolution by flipping the kernel and performing correlation.

    Parameters:
    - input_array: 2D numpy array representing the input matrix.
    - kernel: 2D numpy array representing the kernel/filter.
    - mode: 'valid' or 'full' determines output size and padding behavior.

    Returns:
    - output_array: 2D numpy array result of convolution.
    """
    # Flip the kernel both vertically and horizontally
    flipped_kernel = np.flip(kernel)
    # Call correlate2d with the flipped kernel to compute convolution
    return correlate2d(input_array, flipped_kernel, mode)
