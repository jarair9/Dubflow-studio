"""Audio processing utilities for DubFlow Studio"""

import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import subprocess
import json
from typing import Tuple, Optional

def extract_audio(video_path: str, output_path: str, sample_rate: int = 24000) -> str:
    """Extract audio from video using FFmpeg"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s24le",  # 24-bit PCM for quality
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    return output_path

def get_audio_info(audio_path: str) -> dict:
    """Get audio file information"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", audio_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    # Extract audio stream info
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None
    )
    
    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        "bitrate": int(info.get("format", {}).get("bit_rate", 0))
    }

def separate_stems(audio_path: str, output_dir: str, model: str = "htdemucs") -> Tuple[str, str]:
    """Separate audio into vocals and background using Demucs"""
    import demucs.separate
    
    # Run demucs
    demucs.separate.main(
        ["--two-stems", "vocals", "-n", model, "-o", output_dir, audio_path]
    )
    
    # Find output files
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    model_dir = os.path.join(output_dir, model)
    
    vocals_path = os.path.join(model_dir, base_name, "vocals.wav")
    no_vocals_path = os.path.join(model_dir, base_name, "no_vocals.wav")
    
    return vocals_path, no_vocals_path

def extract_segment(audio_path: str, start: float, end: float, output_path: str) -> str:
    """Extract audio segment"""
    audio = AudioSegment.from_wav(audio_path)
    segment = audio[int(start * 1000):int(end * 1000)]
    segment.export(output_path, format="wav")
    return output_path

def mix_audio_files(
    vocals_path: str,
    bgm_path: str,
    output_path: str,
    vocals_volume: float = 0.0,  # dB
    bgm_volume: float = -10.0,  # dB, ducking
    crossfade_ms: int = 20
) -> str:
    """Mix vocals with background music"""
    vocals = AudioSegment.from_wav(vocals_path)
    bgm = AudioSegment.from_wav(bgm_path)
    
    # Adjust volumes
    vocals = vocals + vocals_volume
    bgm = bgm + bgm_volume
    
    # Match lengths (loop bgm if needed, or truncate)
    if len(bgm) < len(vocals):
        # Loop bgm to match vocals length
        loops = (len(vocals) // len(bgm)) + 1
        bgm = bgm * loops
    
    bgm = bgm[:len(vocals)]
    
    # Mix with crossfade
    mixed = vocals.overlay(bgm, crossfade=crossfade_ms)
    mixed.export(output_path, format="wav")
    
    return output_path

def normalize_lufs(audio_path: str, output_path: str, target_lufs: float = -23.0) -> str:
    """Normalize audio to EBU R128 LUFS standard (broadcast quality)"""
    # Use FFmpeg loudnorm filter
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        "-ar", "24000",
        "-ac", "1",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path

def add_crossfade_between_segments(
    segment_paths: list,
    output_path: str,
    crossfade_ms: int = 20
) -> str:
    """Concatenate segments with crossfades"""
    if not segment_paths:
        return None
    
    # Load first segment
    combined = AudioSegment.from_wav(segment_paths[0])
    
    # Add remaining segments with crossfade
    for path in segment_paths[1:]:
        segment = AudioSegment.from_wav(path)
        combined = combined.append(segment, crossfade=crossfade_ms)
    
    combined.export(output_path, format="wav")
    return output_path

def analyze_audio_quality(audio_path: str) -> dict:
    """Analyze audio quality metrics"""
    y, sr = librosa.load(audio_path, sr=None)
    
    # Signal-to-noise ratio estimation
    # Using a simple approach: high-frequency content as noise proxy
    rms = np.sqrt(np.mean(y**2))
    
    # Dynamic range
    peak = np.max(np.abs(y))
    
    # Spectral centroid (brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Zero crossing rate (noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return {
        "rms": float(rms),
        "peak": float(peak),
        "dynamic_range_db": 20 * np.log10(peak / (rms + 1e-10)),
        "spectral_centroid": float(centroid),
        "zero_crossing_rate": float(zcr),
        "duration": len(y) / sr
    }

def measure_lufs(audio_path: str) -> float:
    """Measure integrated LUFS of audio file"""
    import pyloudnorm as pyln
    
    y, sr = sf.read(audio_path)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    
    return loudness

def apply_deesser(audio_path: str, output_path: str) -> str:
    """Apply de-essing to reduce sibilance"""
    # FFmpeg deesser filter
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", "deesser=i=1.0",
        "-ar", "24000",
        "-ac", "1",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path

def add_breath_pause(audio_path: str, output_path: str, pause_ms: int = 200) -> str:
    """Add breath pause at start of segment"""
    audio = AudioSegment.from_wav(audio_path)
    silence = AudioSegment.silent(duration=pause_ms)
    combined = silence + audio
    combined.export(output_path, format="wav")
    return output_path

def resample_audio(audio_path: str, output_path: str, target_sr: int = 24000) -> str:
    """Resample audio to target sample rate"""
    y, sr = librosa.load(audio_path, sr=target_sr)
    sf.write(output_path, y, target_sr)
    return output_path

def get_segment_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    info = get_audio_info(audio_path)
    return info["duration"]

def count_syllables(text: str, language: str = "en") -> int:
    """Estimate syllable count for text (rough approximation)"""
    if language in ["en", "es", "fr", "de", "it", "pt"]:
        # Simple vowel group counting for Romance/Germanic languages
        vowels = "aeiouyAEIOUY"
        count = 0
        prev_was_vowel = False
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        return max(1, count)
    elif language in ["ja", "zh", "ko"]:
        # For Asian languages, approximate by character count / 2
        return max(1, len(text) // 2)
    else:
        # Default: word-based estimate
        return len(text.split())

def calculate_speaking_rate(text: str, duration: float, language: str = "en") -> float:
    """Calculate syllables per second"""
    syllables = count_syllables(text, language)
    return syllables / duration if duration > 0 else 0
