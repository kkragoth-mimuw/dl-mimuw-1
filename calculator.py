
IMG_W = 250

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w



def calculate_network_sizes(w=IMG_W, layers_count=4, kernel_sizes=[3,5,5, 5]):
  print(f'Init: {w}')
  for layer_index in range(layers_count):
    w_conv, _ = conv_output_shape((w, w), kernel_size=kernel_sizes[layer_index])
    w = (w_conv - 2) / 2 + 1
    print(f'[Layer {layer_index}] post_conv: {w_conv} post_max_pool: {w}')


calculate_network_sizes(w=246, layers_count=4, kernel_sizes=[3,3,5,5])