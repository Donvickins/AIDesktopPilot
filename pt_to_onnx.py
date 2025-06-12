from ultralytics import YOLO

def convert_pt_to_onnx(model_path: str, output_path: str):
    """
    Convert a YOLOv8 PyTorch model to ONNX format.
    
    Args:
        model_path (str): Path to the YOLOv8 PyTorch model file.
        output_path (str): Path where the ONNX model will be saved.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)
    
    # Export the model to ONNX format
    model.export(format='onnx', path=output_path)