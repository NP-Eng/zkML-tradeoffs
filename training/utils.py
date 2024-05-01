import torch 
    
def convert_to_onnx(model, sample_len, onnx_file_path):
    """
    Converts a PyTorch model to the ONNX format and saves it to a specified file path.

    Parameters:
    - model: The PyTorch model to be converted.
    - sample_len: The length of the input sample, specifying the input size.
    - onnx_file_path: The file path where the ONNX model will be saved.

    This function takes a trained PyTorch model and a sample input size, exports the model to the ONNX format,
    and saves it to the provided file path. It specifies model input/output names and handles dynamic batch sizes
    for flexibility in model deployment.
    """
    sample_input = torch.randn(1, sample_len, dtype=torch.float32)

    torch.onnx.export(
        model,  # Model being exported
        sample_input,  # Model input (or a tuple for multiple inputs)
        onnx_file_path,  # Where to save the model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=11,  # ONNX version to export the model to
        do_constant_folding=True,  # Whether to execute constant folding for optimization
        input_names=["input"],  # Model's input names
        output_names=["output"],  # Model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # Variable length axes
            "output": {0: "batch_size"},
        },
    )
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}") 

