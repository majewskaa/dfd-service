import torch
from src.models.xception import XceptionMaxFusionDetector

try:
    print("Attempting to initialize XceptionMaxFusionDetector with pos_freq...")
    model = XceptionMaxFusionDetector(num_classes=2, pos_freq=0.72)
    print("Successfully initialized model with pos_freq=0.72")
    
    # Check if bias was modified (default bias is 0 or random, specific pos_freq sets it to ~0.94)
    # log(0.72 / (1-0.72)) = log(0.72/0.28) = log(2.57) ~= 0.94
    # Check first layer of video model classifier
    if hasattr(model.video_model, 'fc'):
        bias = model.video_model.fc.bias.data
        print(f"Video model fc bias: {bias}")
except TypeError as e:
    print(f"Failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
