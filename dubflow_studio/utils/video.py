"""Video processing utilities for DubFlow Studio"""

import os
import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional

def get_video_info(video_path: str) -> dict:
    """Get video file information using FFprobe"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    
    # Extract video stream info
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None
    )
    
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None
    )
    
    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "format": info.get("format", {}).get("format_name", ""),
        "size_bytes": int(info.get("format", {}).get("size", 0)),
        "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
        "video": {
            "codec": video_stream.get("codec_name") if video_stream else None,
            "width": int(video_stream.get("width", 0)) if video_stream else 0,
            "height": int(video_stream.get("height", 0)) if video_stream else 0,
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if video_stream else 0,
            "pix_fmt": video_stream.get("pix_fmt") if video_stream else None,
        },
        "audio": {
            "codec": audio_stream.get("codec_name") if audio_stream else None,
            "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
            "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        }
    }

def extract_thumbnail(video_path: str, output_path: str, time: float = 0) -> str:
    """Extract thumbnail from video at specified time (seconds)"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(time),  # Seek to time
        "-vframes", "1",  # Extract 1 frame
        "-q:v", "2",  # High quality
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path

def replace_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
    copy_video_codec: bool = True
) -> str:
    """Replace audio track in video"""
    if copy_video_codec:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video without re-encoding
            "-c:a", "aac",  # AAC audio codec
            "-b:a", "192k",  # Audio bitrate
            "-map", "0:v:0",  # Video from first input
            "-map", "1:a:0",  # Audio from second input
            "-shortest",  # End with shortest stream
            "-y",
            output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",  # Re-encode video
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            output_path
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    return output_path

def merge_video_audio_subtitles(
    video_path: str,
    audio_path: str,
    subtitle_path: Optional[str],
    output_path: str
) -> str:
    """Merge video, audio, and optional subtitles"""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-i", audio_path,
    ]
    
    if subtitle_path and os.path.exists(subtitle_path):
        cmd.extend(["-i", subtitle_path])
        subtitle_filter = f"subtitles={subtitle_path}"
        cmd.extend(["-vf", subtitle_filter])
    
    cmd.extend([
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-movflags", "+faststart",  # Web optimization
        "-y",
        output_path
    ])
    
    subprocess.run(cmd, capture_output=True)
    return output_path

def run_wav2lip(
    video_path: str,
    audio_path: str,
    output_path: str,
    model_path: str,
    wav2lip_dir: str,
    quality: str = "gan"
) -> str:
    """Run Wav2Lip for lip synchronization"""
    
    inference_script = os.path.join(wav2lip_dir, "inference.py")
    
    if not os.path.exists(inference_script):
        raise FileNotFoundError(f"Wav2Lip not found at {wav2lip_dir}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Wav2Lip model not found at {model_path}")
    
    cmd = [
        "python", inference_script,
        "--checkpoint_path", model_path,
        "--face", video_path,
        "--audio", audio_path,
        "--outfile", output_path,
    ]
    
    if quality == "gan":
        # Use GAN model settings
        cmd.extend(["--resize_factor", "1"])
    else:
        # Faster, lower quality
        cmd.extend(["--resize_factor", "2"])
    
    # Set environment for imports
    env = os.environ.copy()
    env["PYTHONPATH"] = wav2lip_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip failed: {result.stderr}")
    
    return output_path

def extract_face_crop(
    video_path: str,
    output_path: str,
    face_bbox: Tuple[int, int, int, int]
) -> str:
    """Extract face region from video (for better lip sync)"""
    x, y, w, h = face_bbox
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264",
        "-c:a", "copy",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path

def concatenate_videos(video_paths: list, output_path: str) -> str:
    """Concatenate multiple video files"""
    # Create concat list file
    list_file = output_path + ".txt"
    with open(list_file, "w") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
    
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    os.remove(list_file)
    return output_path

def get_frame_count(video_path: str) -> int:
    """Get total frame count of video"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return 0

def estimate_lip_sync_quality(video_path: str) -> dict:
    """Estimate lip sync quality metrics (placeholder for advanced analysis)"""
    # This would ideally use a face detection + lip movement analysis
    # For now, return basic video quality metrics
    info = get_video_info(video_path)
    
    return {
        "resolution": f"{info['video']['width']}x{info['video']['height']}",
        "fps": info['video']['fps'],
        "duration": info['duration'],
        "quality_score": min(100, int(info['video']['width'] / 10))  # Simple proxy
    }

def create_comparison_video(
    original_path: str,
    dubbed_path: str,
    output_path: str,
    layout: str = "side_by_side"
) -> str:
    """Create side-by-side comparison video"""
    if layout == "side_by_side":
        # Stack horizontally
        filter_complex = "[0:v][1:v]hstack=inputs=2[v]"
    else:
        # Stack vertically
        filter_complex = "[0:v][1:v]vstack=inputs=2[v]"
    
    cmd = [
        "ffmpeg", "-i", original_path,
        "-i", dubbed_path,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a",  # Audio from original
        "-c:v", "libx264",
        "-crf", "23",
        "-y",
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True)
    return output_path
