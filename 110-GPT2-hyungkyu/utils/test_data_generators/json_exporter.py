"""
JSON Exporter for Test Framework

Provides a standard interface to export test data to JSON format.
Works with both NumPy arrays and PyTorch tensors.
"""
import json
import numpy as np


def to_list(data):
    """
    Convert NumPy array or PyTorch tensor to Python list

    Args:
        data: numpy.ndarray or torch.Tensor or list

    Returns:
        Python list representation
    """
    # Check if it's a PyTorch tensor
    if hasattr(data, 'detach'):
        return data.detach().cpu().numpy().tolist()
    # Check if it's a NumPy array
    elif hasattr(data, 'tolist'):
        return data.tolist()
    # Already a list
    else:
        return data


def export_test_data(output_path, input_data, output_data, parameters=None):
    """
    Export test data to JSON format for the test framework

    Args:
        output_path (str): Path to save JSON file
        input_data: Input tensor (numpy array or torch tensor)
        output_data: Expected output tensor (numpy array or torch tensor)
        parameters (dict, optional): Dictionary of parameter tensors {name: data}

    Note:
        Node constructor arguments should be specified in C++ addTest() call,
        not in JSON. JSON only contains input/output/parameters data.

    Example:
        >>> import numpy as np
        >>> from json_exporter import export_test_data
        >>>
        >>> input_data = np.random.randn(2, 3, 8).astype(np.float32)
        >>> output_data = gelu(input_data)
        >>>
        >>> export_test_data(
        ...     output_path="gelu_test.json",
        ...     input_data=input_data,
        ...     output_data=output_data
        ... )
    """
    test_data = {
        "input": to_list(input_data),
        "output": to_list(output_data)
    }

    # Add parameters if provided
    if parameters:
        test_data["parameters"] = {
            name: to_list(value) for name, value in parameters.items()
        }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)

    # Print info
    input_shape = np.array(input_data).shape if not hasattr(input_data, 'shape') else input_data.shape
    output_shape = np.array(output_data).shape if not hasattr(output_data, 'shape') else output_data.shape

    print(f"✓ Test data saved to: {output_path}")
    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: {output_shape}")
    if parameters:
        print(f"  Parameters: {list(parameters.keys())}")
