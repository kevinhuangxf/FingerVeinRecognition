import math
import numpy as np
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    # 4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor,
                           nrow=int(math.sqrt(n_img)),
                           padding=0,
                           normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'
            .format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)

def get_model_params(parameters):
    '''
    Calculate the total params of a model
    
    Args:
        parameters: model.parameters()
    '''

    # Count the total number of parameters
    total_params = sum(p.numel() for p in parameters)

    # Determine the appropriate metric prefix to use
    if total_params >= 1e9:
        prefix = "B"
        divisor = 1e9
    elif total_params >= 1e6:
        prefix = "M"
        divisor = 1e6
    elif total_params >= 1e3:
        prefix = "k"
        divisor = 1e3
    else:
        prefix = ""
        divisor = 1

    # Convert the total number of parameters to the appropriate metric prefix
    total_params /= divisor

    # Print the result
    print(f"Total parameters: {total_params:.1f} {prefix}")
