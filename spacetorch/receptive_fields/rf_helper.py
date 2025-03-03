from collections import OrderedDict
import torch 
import math
from tqdm import tqdm


def find_central_positions_2d(tensor, scale_factor):
    """
    Find central positions in a 2D grid derived from a tensor of shape 
    (batch_size, token_size, embd_size). It assumes token_size can be reshaped
    into a square grid of side length sqrt(token_size - 1).
    
    Parameters:
    - tensor: A tensor of shape (batch_size, token_size, embd_size).
    - scale_factor: The size of the square for which to find central positions.
    
    Returns:
    - A list containing lists of tuples, where each tuple is a central position 
      for each item in the batch.
    """
    batch_size, token_size, _ = tensor.shape
    side_length = int(math.sqrt(token_size - 1))  # Calculate the side length of the square grid
    
    # Calculate central area
    start = (side_length - scale_factor) // 2
    end = start + scale_factor
    
    central_positions_batch = []
    for batch in range(batch_size):
        central_positions = []
        for i in range(start, end):
            for j in range(start, end):
                # Convert 2D position back to 1D + 1 to account for the offset
                pos = 1 + i * side_length + j
                central_positions.append(pos)
        central_positions_batch.append(central_positions)
        
    return central_positions_batch[0]


def sum_central_values(tensor, central_positions_batch):
    """
    Sum up the values at the central positions for each item in the batch.
    
    Parameters:
    - tensor: A tensor of shape (batch_size, token_size, embd_size).
    - central_positions_batch: A list of lists containing central positions for each item in the batch.
    
    Returns:
    - A tensor of shape (batch_size,) where each element is the sum of the central values for the corresponding item.
    """
    # print("tensor",tensor.shape)
    batch_size = tensor.shape[0]
    central_values_sum = torch.zeros(batch_size, dtype=tensor.dtype, device=tensor.device)
    
    for batch_idx, central_positions in enumerate(central_positions_batch):
        for pos in central_positions:
            # Summing up all the values across the embd_size dimension for the central positions
            central_values_sum[batch_idx] += tensor[batch_idx, pos, :].abs().sum()
    # print("central_values_sum", central_values_sum)
    
    return central_values_sum


def find_central_positions_cnn(tensor, scale_factor):
    """
    Finds the central positions of a 4D tensor based on a given scale factor.
    
    Parameters:
    - tensor (Tensor): A 4D tensor with shape (batch_size, channel_size, height, width).
    - scale_factor (int): The number of central positions to find in both height and width.
    
    Returns:
    - A list of tuples, where each tuple represents a central position (width, height).
    """
    _, _, height, width = tensor.shape  # Extract dimensions
    mat = tensor.abs().sum(dim=0).sum(dim=0)
    
    centerh = height // 2
    centerw = width // 2
    
    ph, pw = centerh, centerw
    
    # Calculate the starting and ending indices for central positions
    h_start = max(0, ph - (scale_factor // 2))
    h_end = min(height, ph + (scale_factor // 2))
    if scale_factor == 1 and h_end < height:
        h_end += 1
    w_start = max(0, pw - (scale_factor // 2))
    w_end = min(width, pw + (scale_factor // 2))
    if scale_factor == 1 and w_end < width:
        w_end += 1
    
    # Generate the list of central positions
    central_positions = [(h, w) for h in range(h_start, h_end) for w in range(w_start, w_end)]
    
    return central_positions


def select_layers_based_on_relative_positions(list_of_tuples):
    """
    Select layers from multiple models based on normalized positions, given a list of tuples.
    Each tuple contains a model identifier and a list of layer names.
    
    Parameters:
    - list_of_tuples: A list of tuples. Each tuple contains a model identifier (str)
                      and a list of layer names (list of str).
    
    Returns:
    - A list of tuples. Each tuple contains a model identifier and a list of selected
      layer names based on normalized positions.
    """
    # Step 1: Find the model with the smallest number of layers
    min_length = min(len(layers) for _, layers in list_of_tuples)
    
    # Step 2: Select layers from all lists based on their relative positions
    selected_layers_all_models = []
    
    for model_identifier, layer_list in list_of_tuples:
        num_layers = len(layer_list)
        # Calculate the normalized positions for the current list
        normalized_indices = [round(i * (num_layers - 1) / (min_length - 1)) for i in range(min_length)]
        
        # Ensure indices are within bounds (useful for the last layer if rounding exceeds)
        normalized_indices = [min(i, num_layers - 1) for i in normalized_indices]
        
        # Select layers based on these normalized positions
        selected_layer_names = [layer_list[i] for i in normalized_indices]
        
        # Append the model identifier and selected layers to the output list
        selected_layers_all_models.append((model_identifier, selected_layer_names))
    
    return selected_layers_all_models


def map_layers_to_values(model_layers, models_dict):
    """
    Maps each layer of the given models to its corresponding value using a dictionary of dictionaries.

    Parameters:
    - model_layers (list of tuple): Each tuple contains a model name and a list of layer names.
    - models_dict (dict): Outer dictionary's keys are model names, and each value is another
                          dictionary mapping layer names to numbers.

    Returns:
    - A list of tuples, where each tuple contains a model name, list of layer names, and a list of
      corresponding numbers from the models_dict.
    """
    output_list = []
    
    for model_name, layers in model_layers:
        if model_name in models_dict:
            # Get the dictionary for the current model
            layer_values_dict = models_dict[model_name]
            
            # For each layer, find the corresponding value in the model's dictionary
            layer_values = [layer_values_dict.get(layer, None) for layer in layers]
            
            # Append the model name, layers, and their corresponding values to the output list
            output_list.append((model_name, layers, layer_values))
        else:
            # Model name not found in dictionary, append layers with None values
            output_list.append((model_name, layers, [None] * len(layers)))
    
    return output_list


def find_divisors(n):
    """Find all divisors of a given number n."""
    return [i for i in range(1, n + 1) if n % i == 0]


def update_numbers_to_relative_smallest(input_list):
    """
    For each number in the lists within each tuple, divide it by the smallest number
    found at the same position across all tuples' lists of numbers.

    Parameters:
    - input_list (list of tuples): Each tuple contains a model name, a list of layer names,
      and a list of numbers.

    Returns:
    - A list of tuples where each number list is updated according to the described rule.
    """
    # Determine the smallest number at each position across all tuples
    num_positions = len(input_list[0][2])  # Assuming all numbers lists are of equal length
    smallest_numbers = [min(input_list[j][2][i] for j in range(len(input_list))) for i in range(num_positions)]

    # Update each number by dividing it by the smallest number found at its position
    updated_list = []
    for model_name, layers, numbers in input_list:
        updated_numbers = [numbers[i] / smallest_numbers[i] if smallest_numbers[i] else None for i in range(len(numbers))]
        updated_list.append((model_name, layers, updated_numbers))

    return updated_list


def update_numbers_to_global_smallest(input_list):
    """
    Divide each number in the lists within each tuple by the globally smallest number
    found across all tuples' lists of numbers.

    Parameters:
    - input_list (list of tuples): Each tuple contains a model name, a list of layer names,
      and a list of numbers.

    Returns:
    - A list of tuples where each number list is updated by dividing each number by the globally smallest number.
    """
    # Find the globally smallest number across all tuples' lists of numbers
    global_smallest = min(number for _, _, numbers in input_list for number in numbers)

    # Update each number by dividing it by the global smallest number
    updated_list = []
    for model_name, layers, numbers in input_list:
        updated_numbers = [number / global_smallest for number in numbers]
        updated_list.append((model_name, layers, updated_numbers))

    return updated_list


def analysis_single_layer(model, layer_name, data_loader, hook_dict, max_image_num=None, device="cuda"):
   
    average_image_dict = OrderedDict()
    acummulate_img_index = 0
    model = model.to(device)
    model.eval()
    
    for idx, (images, _) in tqdm(enumerate(data_loader), total=max_image_num):
        if max_image_num is not None and acummulate_img_index >= max_image_num:
            break

        # if idx < max_image_num:
        #     continue

        # if idx > max_image_num:
        #     break
            
        input_tensor = images
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        central_points_scale = 2
        
        model(input_tensor)
            
        k = layer_name
        v = hook_dict[layer_name]

        central_points = find_central_positions_cnn(v, int(central_points_scale))

        v = v.abs().sum(dim=0)
        v = v.sum(dim=0)

        specific_value=0.0
        for (a,b) in central_points:
            specific_value = specific_value+ v[a,b]
        
        specific_value.backward(retain_graph=True)

        input_gradient= input_tensor.grad
        input_gradient= input_gradient.abs().mean(dim=1)
        input_gradient= input_gradient.abs().mean(dim=0,keepdim=True)

        input_tensor.grad.zero_()
        key = f'{k}'

        if key not in average_image_dict:
            average_image_dict[key] = []  # Initialize as a list if the key doesn't exist
            average_image_dict[key].append(input_gradient.cpu().detach().numpy())
        else:
            average_image_dict[key].append(input_gradient.cpu().detach().numpy())
                
        acummulate_img_index = acummulate_img_index + images.shape[0]
    
    return  average_image_dict, hook_dict
