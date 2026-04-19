"""Model loading and caching for DubFlow Studio"""

import streamlit as st
import torch
import os

# Cache resource decorators for model loading
@st.cache_resource
def load_whisper_model(model_size="medium"):
    """Load faster-whisper model"""
    from faster_whisper import WhisperModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model

@st.cache_resource
def load_diarization_model(auth_token=None):
    """Load pyannote.audio diarization model"""
    from pyannote.audio import Pipeline
    
    auth_token = auth_token or os.environ.get("HF_TOKEN")
    if not auth_token:
        return None
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        
        return pipeline
    except Exception as e:
        print(f"Failed to load diarization model: {e}")
        return None

@st.cache_resource
def load_emotion_model():
    """Load emotion detection model"""
    from transformers import pipeline
    
    emotion_classifier = pipeline(
        "audio-classification",
        model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device=0 if torch.cuda.is_available() else -1
    )
    return emotion_classifier

@st.cache_resource
def load_styletts_model():
    """Load StyleTTS 2 model"""
    try:
        import sys
        from pathlib import Path
        
        # StyleTTS 2 installation path
        styletts_path = Path.home() / ".local" / "share" / "StyleTTS2"
        if styletts_path.exists():
            sys.path.insert(0, str(styletts_path))
        
        from styletts2 import StyleTTS2
        
        model = StyleTTS2()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    except Exception as e:
        print(f"StyleTTS 2 not available: {e}")
        return None

@st.cache_resource
def load_xtts_model():
    """Fallback: Load Coqui XTTS v2"""
    from TTS.api import TTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts

class ModelManager:
    """Centralized model management"""
    
    def __init__(self):
        self._models = {}
    
    def get_whisper(self, model_size="medium"):
        """Get or load whisper model"""
        key = f"whisper_{model_size}"
        if key not in self._models:
            self._models[key] = load_whisper_model(model_size)
        return self._models[key]
    
    def get_diarizer(self, auth_token=None):
        """Get or load diarization model"""
        if "diarizer" not in self._models:
            self._models["diarizer"] = load_diarization_model(auth_token)
        return self._models["diarizer"]
    
    def get_emotion_classifier(self):
        """Get or load emotion model"""
        if "emotion" not in self._models:
            self._models["emotion"] = load_emotion_model()
        return self._models["emotion"]
    
    def get_tts_model(self, prefer_styletts=True):
        """Get TTS model, preferring StyleTTS 2"""
        if prefer_styletts and "styletts" not in self._models:
            self._models["styletts"] = load_styletts_model()
        
        if self._models.get("styletts"):
            return self._models["styletts"], "styletts"
        
        if "xtts" not in self._models:
            self._models["xtts"] = load_xtts_model()
        
        return self._models["xtts"], "xtts"
    
    def unload_all(self):
        """Unload all models to free memory"""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance
model_manager = ModelManager()
