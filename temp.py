import torch


file_path = "PredictionModel/plant_disease_model.pth"


data = torch.load(file_path, map_location='cpu')


print("Type of loaded object:", type(data))

# Step 2: If it's a dict, list its keys
if isinstance(data, dict):
    print("\nKeys in the dictionary:")
    for key in data.keys():
        print(" -", key)

# Step 3: If it looks like a state_dict (weights), print layer names and shapes
def inspect_state_dict(state_dict):
    print("\nLayers and their shapes:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tuple(tensor.shape)}")

# Check if it's likely a state_dict
if isinstance(data, dict):
    # Some checkpoints store everything in 'model_state_dict'
    if 'model_state_dict' in data:
        inspect_state_dict(data['model_state_dict'])
    else:
        # If keys look like layer names, treat as state_dict directly
        sample_tensor = next(iter(data.values()))
        if hasattr(sample_tensor, 'shape'):
            inspect_state_dict(data)
elif hasattr(data, 'state_dict'):
    # Full model object
    inspect_state_dict(data.state_dict())
else:
    print("\nLoaded object is neither a dict nor a model with state_dict. It might be a custom object.")
