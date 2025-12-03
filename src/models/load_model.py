import torch
import json
import cv2
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple
from xception import XceptionMaxFusionDetector
from AVFF import AVClassifier

def load_xception_model(model_path: str = r"C:\Users\Mateusz\Desktop\mgr\dfd-lab\cnn_lstm_detector.pth", 
                       info_path: str = r"C:\Users\Mateusz\Desktop\mgr\dfd-lab\cnn_lstm_detector_info.json"):
    """
    Load a saved XceptionMaxFusionDetector model.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        info_path: Path to the model info file (.json file)
    
    Returns:
        Loaded XceptionMaxFusionDetector model
    """
    # Check if files exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(info_path).exists():
        raise FileNotFoundError(f"Model info file not found: {info_path}")
    
    # Load model info
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    print(f"Loading model: {model_info['model_type']}")
    print(f"Number of classes: {model_info['num_classes']}")
    print(f"Description: {model_info['description']}")
    
    # Create model instance
    model = XceptionMaxFusionDetector(
        num_classes=model_info['num_classes']
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    return model

def load_model_with_device(model_path: str = r"C:\Users\Mateusz\Desktop\mgr\dfd-lab\cnn_lstm_detector.pth",
                          info_path: str = r"C:\Users\Mateusz\Desktop\mgr\dfd-lab\cnn_lstm_detector_info.json",
                          device: str = "auto"):
    """
    Load a saved XceptionMaxFusionDetector model with device specification.
    
    Args:
        model_path: Path to the saved model weights (.pth file)
        info_path: Path to the model info file (.json file)
        device: Device to load the model on ("cpu", "cuda", or "auto")
    
    Returns:
        Loaded XceptionMaxFusionDetector model on specified device
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model on device: {device}")
    
    # Load model
    model = load_xception_model(model_path, info_path)
    
    # Move to device
    model = model.to(device)
    
    print(f"Model moved to device: {device}")
    return model

def load_avclassifier_model(encoder_path: str = "pretrained_encoders.pth",
                           info_path: str = "avclassifier_info.json"):
    """
    Load a saved AVClassifier model with pretrained encoders.
    
    Args:
        encoder_path: Path to the saved encoder weights (.pth file)
        info_path: Path to the model info file (.json file)
    
    Returns:
        Loaded AVClassifier model
    """
    # Check if files exist
    if not Path(encoder_path).exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    
    if not Path(info_path).exists():
        raise FileNotFoundError(f"Model info file not found: {info_path}")
    
    # Load model info
    with open(info_path, 'r') as f:
        model_info = json.load(f)
    
    print(f"Loading model: {model_info['model_type']}")
    print(f"Number of classes: {model_info['num_classes']}")
    print(f"Embedding dimension: {model_info['embed_dim']}")
    print(f"Video patch size: {model_info['video_patch']}")
    print(f"Audio patch size: {model_info['audio_patch']}")
    print(f"Number of slices: {model_info['num_slices']}")
    print(f"Freeze encoders: {model_info['freeze_encoders']}")
    print(f"Description: {model_info['description']}")
    
    # Create model instance
    model = AVClassifier(
        num_classes=model_info['num_classes'],
        embed_dim=model_info['embed_dim'],
        video_in_channels=model_info['video_in_channels'],
        audio_in_channels=model_info['audio_in_channels'],
        video_patch=tuple(model_info['video_patch']),
        audio_patch=tuple(model_info['audio_patch']),
        num_slices=model_info['num_slices'],
        encoder_layers=model_info.get('encoder_layers', 2),  # Default to 2 if not specified
        freeze_encoders=model_info['freeze_encoders']
    )
    
    # Load pretrained encoders
    model.load_encoders(encoder_path, device="cpu")
    
    # Set to evaluation mode
    model.eval()
    
    print("AVClassifier model loaded successfully!")
    return model

def load_avclassifier_with_device(encoder_path: str = "pretrained_encoders.pth",
                                 info_path: str = "avclassifier_info.json",
                                 device: str = "auto"):
    """
    Load a saved AVClassifier model with device specification.
    
    Args:
        encoder_path: Path to the saved encoder weights (.pth file)
        info_path: Path to the model info file (.json file)
        device: Device to load the model on ("cpu", "cuda", or "auto")
    
    Returns:
        Loaded AVClassifier model on specified device
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading AVClassifier on device: {device}")
    
    # Load model
    model = load_avclassifier_model(encoder_path, info_path)
    
    model = model.to(device)
    
    print(f"AVClassifier moved to device: {device}")
    return model

def load_video_frames_avff(video_path: str, max_frames: int = 30, target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Load video frames from a video file for AVClassifier (smaller size).
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        target_size: Target size for the frames (height, width)
    
    Returns:
        Tensor of shape (num_frames, 3, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Define transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")
    
    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames)
    print(f"Loaded {len(frames)} frames from video for AVClassifier")
    return frames_tensor

def load_video_frames(video_path: str, max_frames: int = 30, target_size: Tuple[int, int] = (299, 299)) -> torch.Tensor:
    """
    Load video frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        target_size: Target size for the frames (height, width)
    
    Returns:
        Tensor of shape (num_frames, 3, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Define transforms for preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)
        frame_count += 1
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")
    
    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames)
    print(f"Loaded {len(frames)} frames from video")
    return frames_tensor

def create_dummy_spectrogram(target_size: Tuple[int, int] = (299, 299)) -> torch.Tensor:
    """
    Create a dummy spectrogram for testing purposes.
    
    Args:
        target_size: Target size for the spectrogram (height, width)
    
    Returns:
        Tensor of shape (1, 1, height, width) - single spectrogram for CNN-LSTM
    """
    # Create a dummy mel spectrogram-like pattern
    height, width = target_size
    
    # Create a frequency-time pattern that looks like a spectrogram
    freq_axis = np.linspace(0, 1, height)
    time_axis = np.linspace(0, 1, width)
    
    # Create a 2D pattern that resembles a spectrogram
    spectrogram_2d = np.zeros((height, width))
    
    # Add some frequency bands
    for i in range(5):
        freq_center = 0.2 + i * 0.15
        freq_width = 0.05
        freq_mask = np.exp(-((freq_axis - freq_center) ** 2) / (2 * freq_width ** 2))
        
        # Add time-varying intensity
        time_intensity = 0.5 + 0.5 * np.sin(2 * np.pi * (i + 1) * time_axis)
        
        for j in range(width):
            spectrogram_2d[:, j] += freq_mask * time_intensity[j]
    
    # Normalize to [0, 1]
    spectrogram_2d = (spectrogram_2d - spectrogram_2d.min()) / (spectrogram_2d.max() - spectrogram_2d.min())
    
    # Convert to tensor and add channel dimension (single channel for CNN-LSTM)
    spectrogram_tensor = torch.from_numpy(spectrogram_2d).float()
    
    # Convert to 1-channel for CNN-LSTM
    spectrogram_1ch = spectrogram_tensor.unsqueeze(0)  # Shape: (1, height, width)
    
    # Normalize using standard normalization for single channel
    mean = spectrogram_1ch.mean()
    std = spectrogram_1ch.std()
    spectrogram_1ch = (spectrogram_1ch - mean) / (std + 1e-8)
    
    print(f"Created dummy spectrogram of shape: {spectrogram_1ch.shape}")
    return spectrogram_1ch.unsqueeze(0)  # Add batch dimension

def create_dummy_spectrogram_avff(target_size: Tuple[int, int] = (64, 64)) -> torch.Tensor:
    """
    Create a dummy spectrogram for AVClassifier testing purposes.
    
    Args:
        target_size: Target size for the spectrogram (height, width)
    
    Returns:
        Tensor of shape (1, height, width) - single spectrogram for AVClassifier
    """
    # Create a dummy mel spectrogram-like pattern
    height, width = target_size
    
    # Create a frequency-time pattern that looks like a spectrogram
    freq_axis = np.linspace(0, 1, height)
    time_axis = np.linspace(0, 1, width)
    
    # Create a 2D pattern that resembles a spectrogram
    spectrogram_2d = np.zeros((height, width))
    
    # Add some frequency bands
    for i in range(5):
        freq_center = 0.2 + i * 0.15
        freq_width = 0.05
        freq_mask = np.exp(-((freq_axis - freq_center) ** 2) / (2 * freq_width ** 2))
        
        # Add time-varying intensity
        time_intensity = 0.5 + 0.5 * np.sin(2 * np.pi * (i + 1) * time_axis)
        
        for j in range(width):
            spectrogram_2d[:, j] += freq_mask * time_intensity[j]
    
    # Normalize to [0, 1]
    spectrogram_2d = (spectrogram_2d - spectrogram_2d.min()) / (spectrogram_2d.max() - spectrogram_2d.min())
    
    # Convert to tensor
    spectrogram_tensor = torch.from_numpy(spectrogram_2d).float()
    
    print(f"Created dummy spectrogram for AVClassifier of shape: {spectrogram_tensor.shape}")
    return spectrogram_tensor

def predict_video(model, video_path: str, max_frames: int = 10, save_results: bool = True):
    """
    Use the loaded model to predict on a video using the new video sequence methods.
    
    Args:
        model: Loaded XceptionMaxFusionDetector model
        video_path: Path to the video file
        max_frames: Maximum number of frames to process
        save_results: Whether to save prediction results
    """
    print(f"\n=== Predicting on video: {video_path} ===")
    
    # Get the device the model is on
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Load video frames
    try:
        frames = load_video_frames(video_path, max_frames=max_frames, target_size=(299, 299))
        print(f"Frames shape: {frames.shape}")
        # Move frames to the same device as the model
        frames = frames.to(device)
        print(f"Frames moved to device: {device}")
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return
    
    # Create dummy spectrogram (same as in test_xception.py)
    try:
        spectrogram = create_dummy_spectrogram(target_size=(299, 299))
        print(f"Spectrogram shape: {spectrogram.shape}")
        # Move spectrogram to the same device as the model
        spectrogram = spectrogram.to(device)
        print(f"Spectrogram moved to device: {device}")
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare video sequence input: (batch_size=1, num_frames, channels, height, width)
    video_sequence = frames.unsqueeze(0)  # Add batch dimension
    audio_sequence = spectrogram.unsqueeze(0).repeat(1, len(frames), 1, 1, 1)  # Repeat for each frame
    
    print(f"Video sequence shape: {video_sequence.shape}")
    print(f"Audio sequence shape: {audio_sequence.shape}")
    
    # Use the existing predict method which now handles video sequences
    print("\n--- Using Video Sequence Prediction Method ---")
    
    with torch.no_grad():
        # Get overall video prediction
        overall_probs = model.predict(video_sequence, audio_sequence)
        overall_prediction = torch.argmax(overall_probs, dim=1).item()
        overall_confidence = torch.max(overall_probs, dim=1)[0].item()
        
        # Get frame-by-frame predictions by creating individual sequences
        frame_predictions = []
        frame_confidences = []
        
        for i in range(len(frames)):
            # Create single-frame sequence: (batch_size=1, num_frames=1, channels, height, width)
            frame_sequence = frames[i:i+1].unsqueeze(0)  # Shape: (1, 1, 3, 299, 299)
            # spectrogram is already (1, 3, 299, 299), so we just need to add the num_frames dimension
            frame_audio_sequence = spectrogram.unsqueeze(1)  # Shape: (1, 1, 3, 299, 299)
            
            # Get prediction for this frame
            frame_probs = model.predict(frame_sequence, frame_audio_sequence)
            frame_pred = torch.argmax(frame_probs, dim=1).item()
            frame_conf = torch.max(frame_probs, dim=1)[0].item()
            
            frame_predictions.append(frame_pred)
            frame_confidences.append(frame_conf)
        
        # Count frame votes
        fake_count = sum(1 for c in frame_predictions if c == 1)
        real_count = len(frame_predictions) - fake_count
        frame_votes = [real_count, fake_count]
    
    # Print frame-by-frame results
    print("\n--- Frame-by-frame predictions ---")
    for i, (pred, conf) in enumerate(zip(frame_predictions, frame_confidences)):
        class_label = "FAKE" if pred == 1 else "REAL"
        print(f"Frame {i+1}: {class_label} (Confidence: {conf:.4f})")
    
    # Print overall results
    print(f"\n--- Overall Results ---")
    overall_class = "FAKE" if overall_prediction == 1 else "REAL"
    print(f"Overall video classification: {overall_class}")
    print(f"Overall confidence: {overall_confidence:.4f}")
    print(f"Frame votes - REAL: {frame_votes[0]}, FAKE: {frame_votes[1]}")
    
    # Calculate statistics
    fake_count = frame_votes[1]
    real_count = frame_votes[0]
    total_frames = len(frame_predictions)
    avg_confidence = np.mean(frame_confidences)
    
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames classified as FAKE: {fake_count}")
    print(f"Frames classified as REAL: {real_count}")
    print(f"Average confidence: {avg_confidence:.4f}")
    
    if fake_count > real_count:
        fake_percentage = (fake_count / total_frames) * 100
        print(f"Majority classification: {overall_class} ({fake_percentage:.1f}% of frames)")
    elif real_count > fake_count:
        real_percentage = (real_count / total_frames) * 100
        print(f"Majority classification: {overall_class} ({real_percentage:.1f}% of frames)")
    else:
        print("Majority classification: UNCERTAIN (Equal votes)")
    
    # Save results if requested
    if save_results:
        results = {
            'video_path': video_path,
            'overall_classification': overall_class,
            'overall_confidence': overall_confidence,
            'frame_predictions': frame_predictions,
            'frame_confidences': frame_confidences,
            'frame_votes': frame_votes,
            'statistics': {
                'total_frames': total_frames,
                'fake_frames': fake_count,
                'real_frames': real_count,
                'average_confidence': avg_confidence
            }
        }
        
        results_path = Path("prediction_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    return overall_class, frame_predictions, frame_confidences

def predict_video_avff(model, video_path: str, max_frames: int = 8, save_results: bool = True):
    """
    Use the loaded AVClassifier model to predict on a video.
    
    Args:
        model: Loaded AVClassifier model
        video_path: Path to the video file
        max_frames: Maximum number of frames to process
        save_results: Whether to save prediction results
    """
    print(f"\n=== Predicting on video with AVClassifier: {video_path} ===")
    
    # Get the device the model is on
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    # Load video frames (smaller size for AVClassifier)
    try:
        frames = load_video_frames_avff(video_path, max_frames=max_frames, target_size=(64, 64))
        print(f"Frames shape: {frames.shape}")
        # Move frames to the same device as the model
        frames = frames.to(device)
        print(f"Frames moved to device: {device}")
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return
    
    # Create dummy spectrogram for AVClassifier
    try:
        spectrogram = create_dummy_spectrogram_avff(target_size=(64, 64))
        print(f"Spectrogram shape: {spectrogram.shape}")
        # Move spectrogram to the same device as the model
        spectrogram = spectrogram.to(device)
        print(f"Spectrogram moved to device: {device}")
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare input for AVClassifier: (batch_size, num_frames, channels, height, width)
    video_sequence = frames.unsqueeze(0)  # Add batch dimension: (1, num_frames, 3, 64, 64)
    audio_sequence = spectrogram.unsqueeze(0).unsqueeze(0).repeat(1, len(frames), 1, 1, 1)  # (1, num_frames, 1, 64, 64)
    
    print(f"Video sequence shape: {video_sequence.shape}")
    print(f"Audio sequence shape: {audio_sequence.shape}")
    
    print("\n--- Using AVClassifier Prediction Method ---")
    
    with torch.no_grad():
        # Get overall video prediction
        logits = model(video_sequence, audio_sequence)
        overall_prediction = torch.argmax(logits, dim=1).item()
        overall_confidence = torch.softmax(logits, dim=1).max(dim=1)[0].item()
        
        # Get frame-by-frame predictions
        frame_predictions = []
        frame_confidences = []
        
        for i in range(len(frames)):
            # Create sequence with minimum frames needed for 3D patches (patch size is 2)
            if i == 0:
                # First frame: use first two frames
                frame_sequence = frames[0:2].unsqueeze(0)  # Shape: (1, 2, 3, 64, 64)
            elif i == len(frames) - 1:
                # Last frame: use last two frames
                frame_sequence = frames[-2:].unsqueeze(0)  # Shape: (1, 2, 3, 64, 64)
            else:
                # Middle frames: use current frame and next frame
                frame_sequence = frames[i:i+2].unsqueeze(0)  # Shape: (1, 2, 3, 64, 64)
            
            # Create corresponding audio sequence (repeat spectrogram for 2 frames)
            frame_audio_sequence = spectrogram.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1, 1)  # Shape: (1, 2, 1, 64, 64)
            
            # Get prediction for this frame
            frame_logits = model(frame_sequence, frame_audio_sequence)
            frame_pred = torch.argmax(frame_logits, dim=1).item()
            frame_conf = torch.softmax(frame_logits, dim=1).max(dim=1)[0].item()
            
            frame_predictions.append(frame_pred)
            frame_confidences.append(frame_conf)
        
        # Count frame votes
        fake_count = sum(1 for c in frame_predictions if c == 1)
        real_count = len(frame_predictions) - fake_count
        frame_votes = [real_count, fake_count]
    
    # Print frame-by-frame results
    print("\n--- Frame-by-frame predictions ---")
    for i, (pred, conf) in enumerate(zip(frame_predictions, frame_confidences)):
        class_label = "FAKE" if pred == 1 else "REAL"
        print(f"Frame {i+1}: {class_label} (Confidence: {conf:.4f})")
    
    # Print overall results
    print(f"\n--- Overall Results ---")
    overall_class = "FAKE" if overall_prediction == 1 else "REAL"
    print(f"Overall video classification: {overall_class}")
    print(f"Overall confidence: {overall_confidence:.4f}")
    print(f"Frame votes - REAL: {frame_votes[0]}, FAKE: {frame_votes[1]}")
    
    # Calculate statistics
    fake_count = frame_votes[1]
    real_count = frame_votes[0]
    total_frames = len(frame_predictions)
    avg_confidence = np.mean(frame_confidences)
    
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames classified as FAKE: {fake_count}")
    print(f"Frames classified as REAL: {real_count}")
    print(f"Average confidence: {avg_confidence:.4f}")
    
    if fake_count > real_count:
        fake_percentage = (fake_count / total_frames) * 100
        print(f"Majority classification: {overall_class} ({fake_percentage:.1f}% of frames)")
    elif real_count > fake_count:
        real_percentage = (real_count / total_frames) * 100
        print(f"Majority classification: {overall_class} ({real_percentage:.1f}% of frames)")
    else:
        print("Majority classification: UNCERTAIN (Equal votes)")
    
    # Save results if requested
    if save_results:
        results = {
            'model_type': 'AVClassifier',
            'video_path': video_path,
            'overall_classification': overall_class,
            'overall_confidence': overall_confidence,
            'frame_predictions': frame_predictions,
            'frame_confidences': frame_confidences,
            'frame_votes': frame_votes,
            'statistics': {
                'total_frames': total_frames,
                'fake_frames': fake_count,
                'real_frames': real_count,
                'average_confidence': avg_confidence
            }
        }
        
        results_path = Path("avclassifier_prediction_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    return overall_class, frame_predictions, frame_confidences

if __name__ == "__main__":
    # Example usage
    try:
        # Video path
        video_path = r"C:\Users\Mateusz\Desktop\mgr\dfd-lab\data\test.mp4"
        
        # Check if video file exists
        if not Path(video_path).exists():
            print(f"Video file not found: {video_path}")
            print("Please make sure the video file exists at the specified path.")
            print("You can modify the video_path variable in the script to point to your video file.")
            exit(1)
        
        # Try to load AVClassifier first (new model)
        print("=== Trying to load AVClassifier ===")
        try:
            avclassifier_model = load_avclassifier_with_device(device="auto")
            print(f"AVClassifier loaded on device: {next(avclassifier_model.parameters()).device}")
            
            # Predict on the video with AVClassifier
            overall_class, frame_predictions, frame_confidences = predict_video_avff(
                avclassifier_model, 
                video_path, 
                max_frames=8, 
                save_results=True
            )
            
            print(f"\nAVClassifier Result: Video classified as {overall_class}")
            print("Check 'avclassifier_prediction_results.json' for detailed results!")
            
        except FileNotFoundError as e:
            print(f"AVClassifier files not found: {e}")
            print("Please run test_encoder_pretrain.py first to create the pretrained encoders.")
            
            # Fallback to Xception model
            print("\n=== Falling back to Xception Model ===")
            try:
                xception_model = load_model_with_device(device="auto")
                print(f"Xception model loaded on device: {next(xception_model.parameters()).device}")
                
                # Predict on the video with Xception
                overall_class, frame_predictions, frame_confidences = predict_video(
                    xception_model, 
                    video_path, 
                    max_frames=10, 
                    save_results=True
                )
                
                print(f"\nXception Result: Video classified as {overall_class}")
                print("Check 'prediction_results.json' for detailed results!")
                
            except FileNotFoundError as e2:
                print(f"Xception model files not found: {e2}")
                print("Please run test_xception.py first to create the saved model.")
        
        print("\n=== Model Loading and Prediction Test Completed! ===")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the appropriate test scripts first:")
        print("- test_encoder_pretrain.py for AVClassifier")
        print("- test_xception.py for Xception")
        print("Also ensure you have the required dependencies installed (opencv-python, matplotlib).")
