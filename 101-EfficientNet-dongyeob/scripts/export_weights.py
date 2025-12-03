import torch
import torchvision.models as models
from safetensors.torch import save_file
import os

def export_weights():
    os.makedirs("weights", exist_ok=True)
    
    versions = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]
    
    for v in versions:
        print(f"Processing EfficientNet-{v}...")
        try:
            # Load model with default (pretrained) weights
            model_fn = getattr(models, f"efficientnet_{v}")
            weights_enum = getattr(models, f"EfficientNet_{v.upper()}_Weights")
            model = model_fn(weights=weights_enum.DEFAULT)
            
            state_dict = model.state_dict()
            
            # Save to safetensors
            output_path = f"weights/efficientnet-{v}.safetensors"
            save_file(state_dict, output_path)
            print(f"Saved to {output_path}")
            
            # Print keys of B0 for debugging mapping
            if v == "b0":
                print(f"\nKeys for EfficientNet-{v}:")
                for key in state_dict.keys():
                    print(key)
                print("\n")
                
        except Exception as e:
            print(f"Failed to export EfficientNet-{v}: {e}")

if __name__ == "__main__":
    export_weights()
