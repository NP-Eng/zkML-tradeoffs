
import torch    

def convert_to_onnx(model, sample_input, onnx_file_path):
    """
    Converts a PyTorch model to the ONNX format and saves it to a specified file path.
    """
    
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