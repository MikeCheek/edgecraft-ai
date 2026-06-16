import numpy as np
import cv2
import librosa
from typing import Tuple, Optional
from io import BytesIO
from PIL import Image

class DataProcessor:
    """Data processing utilities for TinyML tasks"""
    
    # Standard TinyML input shapes
    IMAGE_SHAPES = {
        "IMAGE_CLASSIFICATION": (224, 224, 3),
        "OBJECT_DETECTION": (224, 224, 3),
        "VISUAL_WAKE_WORDS": (96, 96, 1)  # Grayscale
    }
    
    AUDIO_PARAMS = {
        "KEYWORD_SPOTTING": {
            "sample_rate": 16000,
            "duration": 1.0,
            "n_mfcc": 40
        },
        "AUDIO_CLASSIFICATION": {
            "sample_rate": 16000,
            "duration": 2.0,
            "n_mfcc": 64
        }
    }
    
    @staticmethod
    def preprocess_image(image_data: bytes, task: str) -> np.ndarray:
        """Preprocess image data for the task"""
        img = Image.open(BytesIO(image_data))
        shape = DataProcessor.IMAGE_SHAPES.get(task, (224, 224, 3))
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to target shape
        img = img.resize((shape[0], shape[1]))
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        
        # Handle grayscale for Visual Wake Words
        if shape[2] == 1:
            img_array = cv2.cvtColor((img_array).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            img_array = np.expand_dims(img_array, axis=-1)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        return img_array
    
    @staticmethod
    def preprocess_audio(audio_data: bytes, task: str) -> np.ndarray:
        """Preprocess audio data for the task"""
        # Load audio
        audio, sr = librosa.load(BytesIO(audio_data), sr=None)
        
        params = DataProcessor.AUDIO_PARAMS.get(task, DataProcessor.AUDIO_PARAMS["KEYWORD_SPOTTING"])
        target_sr = params["sample_rate"]
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=params["n_mfcc"])
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
        
        return mfcc
    
    @staticmethod
    def augment_image(image: np.ndarray) -> list:
        """Generate augmented versions of an image"""
        augmented = [image]
        
        # Horizontal flip
        augmented.append(np.fliplr(image))
        
        # Slight rotation (Using cv2 instead of scipy)
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        # Get rotation matrix for 15 degrees, scale 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        # Apply the rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        augmented.append(rotated)
        
        # Brightness adjustment
        bright = np.clip(image * 1.2, 0, 1)
        augmented.append(bright)
        
        return augmented
    
    @staticmethod
    def normalize_batch(batch: np.ndarray) -> np.ndarray:
        """Normalize a batch of images"""
        # Standardize to mean=0, std=1
        mean = np.mean(batch, axis=(0, 1, 2))
        std = np.std(batch, axis=(0, 1, 2))
        
        return (batch - mean) / (std + 1e-9)
