"""Pipeline stages for DubFlow Studio"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

from .config import Config
from .audio import (
    extract_audio, get_audio_info, separate_stems, extract_segment,
    mix_audio_files, normalize_lufs, add_crossfade_between_segments,
    analyze_audio_quality, count_syllables, calculate_speaking_rate,
    get_segment_duration, resample_audio, apply_deesser
)
from .video import (
    get_video_info, replace_audio, merge_video_audio_subtitles,
    run_wav2lip
)
from .models import model_manager

def save_checkpoint(project_dir: str, name: str, data: any):
    """Save pipeline checkpoint"""
    checkpoint_path = Path(project_dir) / f"checkpoint_{name}.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def load_checkpoint(project_dir: str, name: str) -> any:
    """Load pipeline checkpoint"""
    checkpoint_path = Path(project_dir) / f"checkpoint_{name}.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ============================================================================
# STAGE 1: Video Ingest
# ============================================================================
def stage1_ingest(
    video_path: str,
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> Tuple[str, dict]:
    """
    Stage 1: Extract audio and video info
    Returns: (audio_path, video_info)
    """
    if progress_callback:
        progress_callback(0)
    
    # Extract video info
    video_info = get_video_info(video_path)
    
    if progress_callback:
        progress_callback(30)
    
    # Extract audio
    audio_path = os.path.join(project_dir, "audio_extracted.wav")
    extract_audio(video_path, audio_path, sample_rate=Config.SAMPLE_RATE)
    
    if progress_callback:
        progress_callback(70)
    
    # Extract thumbnail
    thumbnail_path = os.path.join(project_dir, "thumbnail.jpg")
    from .video import extract_thumbnail
    extract_thumbnail(video_path, thumbnail_path, time=0)
    
    if progress_callback:
        progress_callback(100)
    
    return audio_path, video_info

# ============================================================================
# STAGE 2: Transcription
# ============================================================================
def stage2_transcribe(
    audio_path: str,
    model_size: str,
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Stage 2: Transcribe audio using faster-whisper
    Returns: List of segments with timestamps
    """
    if progress_callback:
        progress_callback(0)
    
    # Load model
    model = model_manager.get_whisper(model_size)
    
    if progress_callback:
        progress_callback(20)
    
    # Transcribe with VAD
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    if progress_callback:
        progress_callback(60)
    
    # Convert to list format
    segment_list = []
    for seg in segments:
        segment_list.append({
            "id": len(segment_list),
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (seg.words or [])
            ],
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob": seg.no_speech_prob,
            "status": "pending",
            "speaker": "Unknown"
        })
    
    if progress_callback:
        progress_callback(90)
    
    # Save transcript
    transcript_path = os.path.join(project_dir, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segment_list, f, indent=2)
    
    if progress_callback:
        progress_callback(100)
    
    return segment_list

# ============================================================================
# STAGE 3: Speaker Diarization
# ============================================================================
def stage3_diarize(
    audio_path: str,
    segments: List[Dict],
    project_dir: str,
    auth_token: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[Dict], Dict]:
    """
    Stage 3: Perform speaker diarization
    Returns: (updated_segments, speakers_dict)
    """
    if progress_callback:
        progress_callback(0)
    
    # Load diarization model
    pipeline = model_manager.get_diarizer(auth_token)
    
    speakers = {}
    
    if pipeline is None:
        # Fallback: single speaker
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        speakers["SPEAKER_00"] = {"label": "Speaker 0", "segments": []}
        
        if progress_callback:
            progress_callback(100)
        
        return segments, speakers
    
    if progress_callback:
        progress_callback(20)
    
    # Run diarization
    diarization = pipeline(audio_path)
    
    if progress_callback:
        progress_callback(60)
    
    # Map speakers to segments
    for seg in segments:
        seg_start, seg_end = seg["start"], seg["end"]
        seg_duration = seg_end - seg_start
        
        # Find overlapping speakers
        overlaps = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(seg_start, turn.start)
            overlap_end = min(seg_end, turn.end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                overlaps.append((speaker, overlap))
        
        if overlaps:
            # Pick speaker with most overlap
            speaker_totals = {}
            for spk, duration in overlaps:
                speaker_totals[spk] = speaker_totals.get(spk, 0) + duration
            best_speaker = max(speaker_totals, key=speaker_totals.get)
            seg["speaker"] = best_speaker
            
            if best_speaker not in speakers:
                speakers[best_speaker] = {"label": best_speaker, "segments": []}
            speakers[best_speaker]["segments"].append(seg["id"])
        else:
            seg["speaker"] = "SPEAKER_00"
            if "SPEAKER_00" not in speakers:
                speakers["SPEAKER_00"] = {"label": "Speaker 0", "segments": []}
            speakers["SPEAKER_00"]["segments"].append(seg["id"])
    
    if progress_callback:
        progress_callback(80)
    
    # Extract voice samples for each speaker
    samples_dir = os.path.join(project_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for speaker_id in speakers:
        # Find best segment for this speaker (longest, clearest)
        speaker_segs = [segments[i] for i in speakers[speaker_id]["segments"]
                       if i < len(segments)]
        
        if not speaker_segs:
            continue
        
        # Sort by duration
        speaker_segs.sort(key=lambda x: x["end"] - x["start"], reverse=True)
        
        # Pick top segments totaling 6-30 seconds
        selected_duration = 0
        selected_segments = []
        
        for seg in speaker_segs:
            duration = seg["end"] - seg["start"]
            if selected_duration + duration <= Config.MAX_SAMPLE_DURATION:
                selected_segments.append(seg)
                selected_duration += duration
            if selected_duration >= Config.MIN_SAMPLE_DURATION:
                break
        
        if selected_segments:
            # Extract and merge samples
            sample_parts = []
            for seg in selected_segments:
                part_path = os.path.join(samples_dir, f"{speaker_id}_part_{seg['id']}.wav")
                extract_segment(audio_path, seg["start"], seg["end"], part_path)
                sample_parts.append(part_path)
            
            # Merge samples
            from pydub import AudioSegment
            combined = AudioSegment.empty()
            for part in sample_parts:
                combined += AudioSegment.from_wav(part)
            
            sample_path = os.path.join(samples_dir, f"{speaker_id}.wav")
            combined.export(sample_path, format="wav")
            
            speakers[speaker_id]["sample_path"] = sample_path
            speakers[speaker_id]["duration"] = selected_duration
    
    if progress_callback:
        progress_callback(100)
    
    return segments, speakers

# ============================================================================
# STAGE 4: Translation with Cerebras Qwen
# ============================================================================
def call_cerebras_qwen(
    prompt: str,
    api_key: str,
    model: str = "qwen-32b",
    temperature: float = 0.3,
    max_tokens: int = 4096
) -> str:
    """Call Cerebras Qwen API"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional dubbing translator. Provide culturally adapted translations suitable for voice acting."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        f"{Config.CEREBRAS_BASE_URL}/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    response.raise_for_status()
    result = response.json()
    
    return result["choices"][0]["message"]["content"]

def stage4_translate_with_qwen(
    segments: List[Dict],
    source_lang: str,
    target_lang: str,
    project_dir: str,
    cerebras_api_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Stage 4: Translate using Cerebras Qwen with script adaptation
    Returns: Updated segments with translated_text
    """
    if progress_callback:
        progress_callback(0)
    
    if not cerebras_api_key:
        # Fallback: skip translation
        for seg in segments:
            seg["translated_text"] = seg["text"]
        return segments
    
    # Process in batches for efficiency
    batch_size = 10
    total_batches = (len(segments) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(segments))
        batch = segments[start_idx:end_idx]
        
        # Build batch prompt
        batch_texts = []
        for i, seg in enumerate(batch):
            duration = seg["end"] - seg["start"]
            syllable_target = int(duration * 3.5)  # ~3.5 syllables/sec
            batch_texts.append(f"{i}. [{seg['speaker']}] [{duration:.1f}s/~{syllable_target} syllables] {seg['text']}")
        
        prompt = f"""Translate these dialogue lines from {Config.LANGUAGES.get(source_lang, source_lang)} to {Config.LANGUAGES.get(target_lang, target_lang)}.

The translations should be:
1. NATURAL and conversational (not literal)
2. CULTURALLY ADAPTED (change idioms/references to local equivalents)
3. TIMING-APPROPRIATE (match the approximate syllable count for lip-sync)
4. EMOTIONALLY CONSISTENT with the speaker

Lines to translate:
{chr(10).join(batch_texts)}

Return ONLY a JSON array in this exact format:
[{{"translation": "...", "syllable_count": N, "adaptation_note": "..."}}]
"""
        
        try:
            response = call_cerebras_qwen(prompt, cerebras_api_key)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                translations = json.loads(json_match.group())
            else:
                translations = json.loads(response)
            
            # Apply translations
            for i, trans in enumerate(translations):
                if start_idx + i < len(segments):
                    seg = segments[start_idx + i]
                    seg["translated_text"] = trans.get("translation", seg["text"])
                    seg["target_syllables"] = trans.get("syllable_count", count_syllables(seg["translated_text"], target_lang))
                    seg["adaptation_note"] = trans.get("adaptation_note", "")
                    
        except Exception as e:
            print(f"Translation batch {batch_idx} failed: {e}")
            # Fallback: copy original
            for i in range(start_idx, end_idx):
                if i < len(segments):
                    segments[i]["translated_text"] = segments[i]["text"]
        
        if progress_callback:
            progress = int((batch_idx + 1) / total_batches * 100)
            progress_callback(progress)
    
    # Save translations
    trans_path = os.path.join(project_dir, "translations.json")
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump([{
            "id": s["id"],
            "original": s["text"],
            "translated": s.get("translated_text", ""),
            "syllables": s.get("target_syllables", 0)
        } for s in segments], f, indent=2)
    
    return segments

# ============================================================================
# STAGE 5: Emotion Detection
# ============================================================================
def stage5_detect_emotions(
    audio_path: str,
    segments: List[Dict],
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Stage 5: Detect emotion per segment
    Returns: Updated segments with emotion field
    """
    if progress_callback:
        progress_callback(0)
    
    try:
        classifier = model_manager.get_emotion_classifier()
    except:
        # Fallback: neutral
        for seg in segments:
            seg["emotion"] = "neutral"
        return segments
    
    total = len(segments)
    
    for idx, seg in enumerate(segments):
        try:
            # Extract segment audio
            seg_path = os.path.join(project_dir, "temp_emotion.wav")
            extract_segment(audio_path, seg["start"], seg["end"], seg_path)
            
            # Predict emotion
            results = classifier(seg_path)
            
            if results:
                # Get top emotion
                top = max(results, key=lambda x: x["score"])
                seg["emotion"] = top["label"].lower()
                seg["emotion_confidence"] = top["score"]
            else:
                seg["emotion"] = "neutral"
                seg["emotion_confidence"] = 0.0
                
        except Exception as e:
            print(f"Emotion detection failed for segment {idx}: {e}")
            seg["emotion"] = "neutral"
            seg["emotion_confidence"] = 0.0
        
        if progress_callback:
            progress = int((idx + 1) / total * 100)
            progress_callback(progress)
    
    return segments

# ============================================================================
# STAGE 6: Voice Cloning and Dubbing (StyleTTS 2 with Duration Control)
# ============================================================================
def stage6_clone_and_dub_styletts(
    segments: List[Dict],
    speakers: Dict,
    target_lang: str,
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Stage 6: Generate dubbed audio using StyleTTS 2
    StyleTTS 2 supports duration conditioning - no time-stretching needed!
    
    Returns: Updated segments with dubbed_audio_path
    """
    if progress_callback:
        progress_callback(0)
    
    # Setup output directory
    dubbed_dir = os.path.join(project_dir, "dubbed_segments")
    os.makedirs(dubbed_dir, exist_ok=True)
    
    # Try StyleTTS 2 first, fall back to XTTS
    tts_model, tts_type = model_manager.get_tts_model(prefer_styletts=True)
    
    total = len(segments)
    
    for idx, seg in enumerate(segments):
        try:
            speaker_id = seg.get("speaker", "SPEAKER_00")
            speaker_info = speakers.get(speaker_id, {})
            sample_path = speaker_info.get("sample_path")
            
            if not sample_path or not os.path.exists(sample_path):
                print(f"No voice sample for speaker {speaker_id}")
                seg["dubbed_audio_path"] = None
                continue
            
            text = seg.get("translated_text", seg.get("text", ""))
            target_duration = seg["end"] - seg["start"]
            emotion = seg.get("emotion", "neutral")
            
            output_path = os.path.join(dubbed_dir, f"seg_{seg['id']:04d}.wav")
            
            if tts_type == "styletts" and tts_model is not None:
                # Use StyleTTS 2 with duration control
                _generate_styletts(
                    tts_model, text, sample_path, output_path,
                    target_duration=target_duration,
                    emotion=emotion,
                    language=target_lang
                )
            else:
                # Fallback: XTTS v2 (no duration control - will need alignment)
                _generate_xtts(
                    tts_model, text, sample_path, output_path,
                    emotion=emotion,
                    language=target_lang
                )
            
            seg["dubbed_audio_path"] = output_path
            
            # Verify generation
            if os.path.exists(output_path):
                seg["dub_status"] = "generated"
                seg["dub_quality"] = analyze_audio_quality(output_path)
            else:
                seg["dub_status"] = "failed"
                
        except Exception as e:
            print(f"Dubbing failed for segment {seg['id']}: {e}")
            seg["dubbed_audio_path"] = None
            seg["dub_status"] = "failed"
        
        if progress_callback:
            progress = int((idx + 1) / total * 100)
            progress_callback(progress)
    
    return segments

def _generate_styletts(
    model,
    text: str,
    sample_path: str,
    output_path: str,
    target_duration: float,
    emotion: str = "neutral",
    language: str = "en"
):
    """Generate audio with StyleTTS 2 (duration-conditioned)"""
    try:
        # StyleTTS 2 generation with length control
        # The model can generate at specific durations
        wav = model.inference(
            text=text,
            ref_audio_path=sample_path,
            target_duration=target_duration,  # Key feature!
            emotion=emotion,
            alpha=0.3,  # Style strength
            beta=0.7,   # Prosody strength
        )
        
        import soundfile as sf
        sf.write(output_path, wav, Config.SAMPLE_RATE)
        
    except Exception as e:
        print(f"StyleTTS generation failed: {e}, falling back to XTTS")
        raise

def _generate_xtts(
    model,
    text: str,
    sample_path: str,
    output_path: str,
    emotion: str = "neutral",
    language: str = "en"
):
    """Generate audio with XTTS v2 (fallback)"""
    model.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=sample_path,
        language=language,
        split_sentences=True
    )

# ============================================================================
# STAGE 7: Separate Audio Stems
# ============================================================================
def stage7_separate_stems(
    audio_path: str,
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> Tuple[str, str]:
    """
    Stage 7: Separate vocals from background
    Returns: (vocals_path, no_vocals_path)
    """
    if progress_callback:
        progress_callback(0)
    
    stems_dir = os.path.join(project_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    
    try:
        import demucs.separate
        
        if progress_callback:
            progress_callback(20)
        
        # Run demucs separation
        demucs.separate.main([
            "--two-stems", "vocals",
            "-n", Config.DEMUCS_MODEL,
            "-o", stems_dir,
            audio_path
        ])
        
        if progress_callback:
            progress_callback(80)
        
        # Find output files
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        model_dir = os.path.join(stems_dir, Config.DEMUCS_MODEL, base_name)
        
        vocals_path = os.path.join(model_dir, "vocals.wav")
        no_vocals_path = os.path.join(model_dir, "no_vocals.wav")
        
        if progress_callback:
            progress_callback(100)
        
        return vocals_path, no_vocals_path
        
    except Exception as e:
        print(f"Demucs separation failed: {e}")
        # Fallback: return original as vocals, no BGM
        return audio_path, None

# ============================================================================
# STAGE 8: Align Audio (Minimal - StyleTTS handles duration)
# ============================================================================
def stage8_align_audio(
    segments: List[Dict],
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Stage 8: Assemble full dubbed audio track
    Since StyleTTS generates at target duration, minimal alignment needed
    Returns: full_dubbed_audio_path
    """
    if progress_callback:
        progress_callback(0)
    
    from pydub import AudioSegment
    
    dubbed_dir = os.path.join(project_dir, "dubbed_segments")
    
    # Sort segments by time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    
    # Find end time for canvas
    if sorted_segments:
        last_end = max(s["end"] for s in sorted_segments)
        canvas_duration = int(last_end * 1000) + 1000  # ms + padding
    else:
        canvas_duration = 1000
    
    # Create silent canvas
    canvas = AudioSegment.silent(duration=canvas_duration)
    
    # Overlay each segment at correct position
    for idx, seg in enumerate(sorted_segments):
        dub_path = seg.get("dubbed_audio_path")
        if dub_path and os.path.exists(dub_path):
            try:
                seg_audio = AudioSegment.from_wav(dub_path)
                position = int(seg["start"] * 1000)
                canvas = canvas.overlay(seg_audio, position=position)
            except Exception as e:
                print(f"Failed to overlay segment {seg['id']}: {e}")
        
        if progress_callback:
            progress = int((idx + 1) / len(sorted_segments) * 100)
            progress_callback(progress)
    
    # Apply crossfades between segments if close together
    # (Already handled by canvas overlay with gaps)
    
    # Export
    full_dubbed_path = os.path.join(project_dir, "dubbed_full.wav")
    canvas.export(full_dubbed_path, format="wav")
    
    if progress_callback:
        progress_callback(100)
    
    return full_dubbed_path

# ============================================================================
# STAGE 9: Mix Audio with Broadcast Standards
# ============================================================================
def stage9_mix_audio(
    dubbed_audio_path: str,
    bgm_path: Optional[str],
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Stage 9: Mix dubbed vocals with background music
    Apply broadcast standards (EBU R128)
    Returns: mixed_audio_path
    """
    if progress_callback:
        progress_callback(0)
    
    # Normalize dubbed audio to broadcast standard
    normalized_path = os.path.join(project_dir, "dubbed_normalized.wav")
    normalize_lufs(dubbed_audio_path, normalized_path, target_lufs=Config.TARGET_LUFS)
    
    if progress_callback:
        progress_callback(30)
    
    # Apply de-esser
    deessed_path = os.path.join(project_dir, "dubbed_deessed.wav")
    apply_deesser(normalized_path, deessed_path)
    
    if progress_callback:
        progress_callback(50)
    
    if bgm_path and os.path.exists(bgm_path):
        # Mix with BGM
        mixed_path = os.path.join(project_dir, "mixed_audio.wav")
        
        # Use smart ducking: BGM at -14dB when speech present
        mix_audio_files(
            deessed_path,
            bgm_path,
            mixed_path,
            vocals_volume=0.0,  # Dub at normal level
            bgm_volume=-12.0,   # Duck BGM
            crossfade_ms=Config.DEFAULT_CROSSFADE_MS
        )
        
        if progress_callback:
            progress_callback(80)
        
        # Final normalization of mix
        final_path = os.path.join(project_dir, "final_audio.wav")
        normalize_lufs(mixed_path, final_path, target_lufs=Config.TARGET_LUFS)
        
    else:
        # No BGM, just use normalized dubbed audio
        final_path = os.path.join(project_dir, "final_audio.wav")
        normalize_lufs(deessed_path, final_path, target_lufs=Config.TARGET_LUFS)
    
    if progress_callback:
        progress_callback(100)
    
    return final_path

# ============================================================================
# STAGE 10: Lip Synchronization
# ============================================================================
def stage10_lipsync(
    video_path: str,
    audio_path: str,
    project_dir: str,
    quality: str = "studio",
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Stage 10: Synchronize lips to dubbed audio
    Returns: lipsynced_video_path
    """
    if progress_callback:
        progress_callback(0)
    
    output_path = os.path.join(project_dir, "lipsynced.mp4")
    
    # Determine model to use
    lip_model = Config.LIPSYNC_MODELS.get(quality, "wav2lip_gan")
    
    if lip_model == "wav2lip" or lip_model == "wav2lip_gan":
        try:
            # Wav2Lip paths
            wav2lip_dir = os.path.join(os.path.dirname(__file__), "..", "Wav2Lip")
            model_file = "wav2lip_gan.pth" if lip_model == "wav2lip_gan" else "wav2lip.pth"
            model_path = os.path.join(wav2lip_dir, "checkpoints", model_file)
            
            if progress_callback:
                progress_callback(20)
            
            # Run Wav2Lip
            run_wav2lip(
                video_path,
                audio_path,
                output_path,
                model_path,
                wav2lip_dir,
                quality=lip_model
            )
            
            if progress_callback:
                progress_callback(100)
            
            return output_path
            
        except Exception as e:
            print(f"Lip sync failed: {e}")
            # Fallback: just replace audio
            return video_path
    
    # For studio quality, would use IP_LAP here if available
    # For now, fallback
    return video_path

# ============================================================================
# STAGE 11: Final Render
# ============================================================================
def stage11_render(
    video_path: str,
    audio_path: str,
    segments: List[Dict],
    project_dir: str,
    progress_callback: Optional[Callable] = None
) -> Tuple[str, str]:
    """
    Stage 11: Render final video with subtitles
    Returns: (final_video_path, srt_path)
    """
    if progress_callback:
        progress_callback(0)
    
    # Generate SRT subtitles
    srt_path = os.path.join(project_dir, "subtitles.srt")
    _generate_srt(segments, srt_path)
    
    if progress_callback:
        progress_callback(30)
    
    # Render final video
    final_path = os.path.join(project_dir, "final_output.mp4")
    
    # Replace audio in video
    replace_audio(video_path, audio_path, final_path)
    
    if progress_callback:
        progress_callback(100)
    
    return final_path, srt_path

def _generate_srt(segments: List[Dict], output_path: str):
    """Generate SRT subtitle file from segments"""
    def format_time(seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            text = seg.get("translated_text", seg.get("text", ""))
            if not text.strip():
                continue
            
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n")
            f.write("\n")
