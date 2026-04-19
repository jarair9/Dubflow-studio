"""Configuration for DubFlow Studio"""

import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    PROJECTS_DIR = BASE_DIR / "projects"
    MODELS_DIR = BASE_DIR / "models"
    
    # Ensure directories exist
    PROJECTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Audio settings
    SAMPLE_RATE = 24000
    TARGET_LUFS = -23.0  # EBU R128 broadcast standard
    
    # Whisper settings
    WHISPER_MODELS = {
        "fast": "small",
        "balanced": "medium", 
        "studio": "large-v3"
    }
    
    # Supported languages
    LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "zh": "Chinese",
        "hi": "Hindi",
        "ur": "Urdu",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic"
    }
    
    # StyleTTS 2 settings
    STYLETTS_MODEL = "SabrinaCarpenter/StyleTTS2-LibriTTS"
    
    # Cerebras settings
    CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
    CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"  # or other available model
    
    # Pipeline settings
    ENABLE_CHECKPOINTING = True
    CHECKPOINT_INTERVAL = 1  # Save after every stage
    
    # Quality settings
    DEFAULT_CROSSFADE_MS = 20  # Crossfade between segments
    DEFAULT_BREATH_PAUSE_MS = 200  # Breath insertion pause
    
    # Lip sync models
    LIPSYNC_MODELS = {
        "fast": "wav2lip",
        "balanced": "wav2lip_gan",
        "studio": "ip_lap"  # If available
    }
    
    # Emotion detection model
    EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    
    # Demucs settings
    DEMUCS_MODEL = "htdemucs"
    DEMUCS_SEGMENT_LENGTH = 10
    
    # Voice cloning settings
    MIN_SAMPLE_DURATION = 3.0  # Minimum seconds for voice sample
    MAX_SAMPLE_DURATION = 30.0  # Maximum seconds for voice sample
    VOICE_EMBEDDING_DIM = 256
