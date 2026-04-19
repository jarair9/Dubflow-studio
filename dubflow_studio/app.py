"""
DubFlow Studio - Professional AI Dubbing Application
Single-file Streamlit app with modular pipeline stages
"""

import streamlit as st
import os
import sys
import json
import threading
import time
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.append(os.path.dirname(__file__))

from utils.config import Config
from utils.pipeline import (
    stage1_ingest, stage2_transcribe, stage3_diarize,
    stage4_translate_with_qwen, stage5_detect_emotions,
    stage6_clone_and_dub_styletts, stage7_separate_stems,
    stage8_align_audio, stage9_mix_audio, stage10_lipsync,
    stage11_render, save_checkpoint, load_checkpoint
)
from utils.state_manager import StateManager

# Page config
st.set_page_config(
    page_title="DubFlow Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
StateManager.init_session_state()

# Sidebar navigation
st.sidebar.title("🎬 DubFlow Studio")
page = st.sidebar.radio(
    "Navigate",
    ["📤 Upload & Settings", "⚙️ Pipeline Monitor", "📝 Script Editor", "👁️ Preview & Export"]
)

# Page 1: Upload & Settings
if page == "📤 Upload & Settings":
    st.title("📤 Upload & Project Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Video Upload")
        video_file = st.file_uploader(
            "Upload video file",
            type=['mp4', 'mkv', 'mov', 'avi'],
            help="Supported formats: MP4, MKV, MOV, AVI"
        )
        
        if video_file:
            st.video(video_file)
    
    with col2:
        st.subheader("Dubbing Settings")
        
        project_name = st.text_input(
            "Project Name",
            value=f"dub_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        source_lang = st.selectbox(
            "Source Language",
            ["en", "es", "fr", "de", "it", "ja", "ko", "zh", "hi", "ur"],
            format_func=lambda x: {
                "en": "English", "es": "Spanish", "fr": "French",
                "de": "German", "it": "Italian", "ja": "Japanese",
                "ko": "Korean", "zh": "Chinese", "hi": "Hindi", "ur": "Urdu"
            }.get(x, x)
        )
        
        target_lang = st.selectbox(
            "Target Language",
            ["hi", "ur", "en", "es", "fr", "de", "ja", "ko", "zh", "it", "pt", "ru", "ar"],
            format_func=lambda x: {
                "en": "English", "es": "Spanish", "fr": "French",
                "de": "German", "it": "Italian", "ja": "Japanese",
                "ko": "Korean", "zh": "Chinese", "hi": "Hindi", "ur": "Urdu",
                "pt": "Portuguese", "ru": "Russian", "ar": "Arabic"
            }.get(x, x)
        )
        
        quality_mode = st.selectbox(
            "Quality Mode",
            ["fast", "balanced", "studio"],
            format_func=lambda x: {
                "fast": "Fast (small models, basic lipsync)",
                "balanced": "Balanced (medium quality)",
                "studio": "Studio (large models, GAN lipsync)"
            }.get(x, x)
        )
        
        # Cerebras Qwen settings
        st.subheader("🤖 Cerebras Qwen Settings")
        use_cerebras = st.checkbox("Use Cerebras Qwen for Translation", value=True)
        
        if use_cerebras:
            cerebras_api_key = st.text_input(
                "Cerebras API Key",
                type="password",
                value=st.session_state.get("cerebras_api_key", "")
            )
            st.session_state.cerebras_api_key = cerebras_api_key
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            enable_lipsync = st.checkbox("Enable Lip Sync", value=True)
            keep_bgm = st.checkbox("Keep Background Music", value=True)
            enable_emotion = st.checkbox("Enable Emotion Detection", value=True)
            auto_retry = st.checkbox("Auto-retry failed stages", value=True)
    
    # Start button
    st.divider()
    if st.button("🚀 Start Dubbing Pipeline", type="primary", use_container_width=True):
        if not video_file:
            st.error("Please upload a video file first!")
        elif use_cerebras and not st.session_state.get("cerebras_api_key"):
            st.error("Please provide Cerebras API Key!")
        else:
            # Initialize project
            project_id = project_name.replace(" ", "_")
            project_dir = os.path.join(Config.PROJECTS_DIR, project_id)
            os.makedirs(project_dir, exist_ok=True)
            
            # Save uploaded video
            video_path = os.path.join(project_dir, "original_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_file.getvalue())
            
            # Update session state
            st.session_state.project_id = project_id
            st.session_state.project_dir = project_dir
            st.session_state.video_path = video_path
            st.session_state.source_lang = source_lang
            st.session_state.target_lang = target_lang
            st.session_state.quality_mode = quality_mode
            st.session_state.use_cerebras = use_cerebras
            st.session_state.enable_lipsync = enable_lipsync
            st.session_state.keep_bgm = keep_bgm
            st.session_state.enable_emotion = enable_emotion
            
            # Initialize pipeline status
            stages = [
                "ingest", "transcribe", "diarize", "translate",
                "emotion", "clone_dub", "separate", "align",
                "mix", "lipsync", "render"
            ]
            for stage in stages:
                st.session_state.pipeline_status[stage] = "pending"
                st.session_state.pipeline_progress[stage] = 0
            
            st.session_state.current_page = "⚙️ Pipeline Monitor"
            st.success("✅ Project initialized! Redirecting to Pipeline Monitor...")
            time.sleep(1)
            st.rerun()

# Page 2: Pipeline Monitor
elif page == "⚙️ Pipeline Monitor":
    st.title("⚙️ Pipeline Monitor")
    
    if not st.session_state.get("project_id"):
        st.warning("No active project. Please start from Upload & Settings page.")
        st.stop()
    
    # Display project info
    st.info(f"**Project:** {st.session_state.project_id} | **Target:** {st.session_state.target_lang}")
    
    # Pipeline stages with status
    stages = [
        ("ingest", "📥 Ingesting Video", "Extracting audio and video info"),
        ("transcribe", "🎤 Transcribing Audio", "faster-whisper transcription"),
        ("diarize", "👥 Detecting Speakers", "pyannote.audio diarization"),
        ("translate", "🌐 Translating Script", "Cerebras Qwen translation"),
        ("emotion", "😊 Analyzing Emotions", "Emotion detection per segment"),
        ("clone_dub", "🎙️ Cloning Voices & Dubbing", "StyleTTS 2 voice cloning"),
        ("separate", "🎵 Separating Audio Stems", "Demucs source separation"),
        ("align", "⏱️ Aligning Audio", "Duration matching"),
        ("mix", "🎚️ Mixing Audio", "Mixing dubbed vocals + BGM"),
        ("lipsync", "👄 Syncing Lips", "Wav2Lip/IP_LAP lip sync"),
        ("render", "🎬 Rendering Final Video", "FFmpeg final assembly")
    ]
    
    # Check if pipeline is running
    pipeline_running = st.session_state.get("pipeline_thread") and st.session_state.pipeline_thread.is_alive()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Pipeline", disabled=pipeline_running, use_container_width=True):
            # Start pipeline in thread
            def run_pipeline():
                try:
                    execute_pipeline()
                except Exception as e:
                    st.session_state.error_msg = str(e)
                    import traceback
                    st.session_state.error_traceback = traceback.format_exc()
            
            thread = threading.Thread(target=run_pipeline, daemon=True)
            st.session_state.pipeline_thread = thread
            thread.start()
            st.rerun()
    
    with col2:
        failed_stages = [s for s, status in st.session_state.pipeline_status.items() if status == "failed"]
        if st.button("🔄 Retry Failed", disabled=not failed_stages or pipeline_running, use_container_width=True):
            for stage in failed_stages:
                st.session_state.pipeline_status[stage] = "pending"
            st.rerun()
    
    with col3:
        if st.button("⏹️ Stop", disabled=not pipeline_running, use_container_width=True):
            # Note: Thread killing is messy in Python
            st.session_state.pipeline_status["current"] = "stopped"
            st.warning("Pipeline stop requested. Please wait for current stage to complete.")
    
    # Progress overview
    total_stages = len(stages)
    completed = sum(1 for s, _ in stages if st.session_state.pipeline_status.get(s[0]) == "complete")
    failed = sum(1 for s, _ in stages if st.session_state.pipeline_status.get(s[0]) == "failed")
    progress_pct = completed / total_stages
    
    st.progress(progress_pct, text=f"Pipeline Progress: {completed}/{total_stages} stages complete")
    
    # Individual stage status
    st.subheader("Stage Status")
    
    for stage_id, stage_name, description in stages:
        status = st.session_state.pipeline_status.get(stage_id, "pending")
        progress = st.session_state.pipeline_progress.get(stage_id, 0)
        
        # Status icons
        icons = {
            "pending": "⏳",
            "running": "🔄",
            "complete": "✅",
            "failed": "❌"
        }
        
        with st.expander(f"{icons.get(status, '⏳')} {stage_name}", expanded=(status == "running")):
            st.caption(description)
            st.progress(progress / 100, text=f"{progress}%")
            
            if status == "failed" and st.session_state.get(f"error_{stage_id}"):
                st.error(st.session_state.get(f"error_{stage_id}"))
    
    # Live logs
    st.subheader("Live Logs")
    log_container = st.container(height=300)
    
    log_path = os.path.join(st.session_state.get("project_dir", ""), "pipeline_log.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = f.read()
            log_container.code(logs[-5000:] if len(logs) > 5000 else logs, language="log")
    
    # Auto-refresh
    if pipeline_running:
        time.sleep(2)
        st.rerun()
    
    # Error display
    if st.session_state.get("error_msg"):
        st.error(f"**Error:** {st.session_state.error_msg}")
        with st.expander("Full Traceback"):
            st.code(st.session_state.get("error_traceback", ""))

# Page 3: Script Editor
elif page == "📝 Script Editor":
    st.title("📝 Script Editor")
    
    if not st.session_state.get("segments"):
        st.warning("No segments available. Complete transcription first.")
        st.stop()
    
    segments = st.session_state.segments
    speakers = st.session_state.get("speakers", {})
    
    # Speaker info
    st.subheader("Detected Speakers")
    for spk_id, spk_info in speakers.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.audio(spk_info.get("sample_path"), format="audio/wav")
        with col2:
            st.write(f"**{spk_id}** - Sample duration: {spk_info.get('duration', 0):.2f}s")
    
    st.divider()
    
    # Segment editor
    st.subheader("Edit Segments")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_speaker = st.selectbox("Filter by Speaker", ["All"] + list(speakers.keys()))
    with col2:
        filter_status = st.selectbox("Filter by Status", ["All", "approved", "needs_review", "modified"])
    with col3:
        search_text = st.text_input("Search text")
    
    # Create editable dataframe
    import pandas as pd
    
    df_data = []
    for i, seg in enumerate(segments):
        status = seg.get("status", "pending")
        color = {"approved": "🟢", "needs_review": "🔴", "modified": "🟡", "pending": "⚪"}.get(status, "⚪")
        
        df_data.append({
            "#": i,
            "Speaker": seg.get("speaker", "Unknown"),
            "Start": f"{seg['start']:.2f}s",
            "End": f"{seg['end']:.2f}s",
            "Original": seg.get("text", "")[:50] + "..." if len(seg.get("text", "")) > 50 else seg.get("text", ""),
            "Translated": seg.get("translated_text", "")[:50] + "..." if len(seg.get("translated_text", "")) > 50 else seg.get("translated_text", ""),
            "Emotion": seg.get("emotion", "neutral"),
            "Status": f"{color} {status}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Filters
    if filter_speaker != "All":
        df = df[df["Speaker"] == filter_speaker]
    if filter_status != "All":
        df = df[df["Status"].str.contains(filter_status)]
    if search_text:
        df = df[df["Original"].str.contains(search_text, case=False) | df["Translated"].str.contains(search_text, case=False)]
    
    # Show editable table
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=["#", "Start", "End", "Original"],
        column_config={
            "Translated": st.column_config.TextColumn("Translated Text", width="large"),
            "Emotion": st.column_config.SelectboxColumn(
                "Emotion",
                options=["neutral", "happy", "sad", "angry", "surprised", "excited"]
            )
        }
    )
    
    # Apply changes button
    if st.button("💾 Save Changes"):
        # Update segments from edited dataframe
        for idx, row in edited_df.iterrows():
            seg_idx = row["#"]
            segments[seg_idx]["translated_text"] = row["Translated"]
            segments[seg_idx]["emotion"] = row["Emotion"]
            if "modified" in str(row["Status"]):
                segments[seg_idx]["status"] = "modified"
        
        st.session_state.segments = segments
        save_checkpoint(st.session_state.project_dir, "segments", segments)
        st.success("Changes saved!")
    
    # Re-dub buttons
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Re-dub Modified Segments"):
            modified_idxs = [i for i, s in enumerate(segments) if s.get("status") == "modified"]
            if modified_idxs:
                st.session_state.redub_queue = modified_idxs
                st.info(f"Queued {len(modified_idxs)} segments for re-dubbing")
            else:
                st.warning("No modified segments to re-dub")
    
    with col2:
        if st.button("✅ Approve All"):
            for seg in segments:
                seg["status"] = "approved"
            st.session_state.segments = segments
            st.success("All segments approved!")

# Page 4: Preview & Export
elif page == "👁️ Preview & Export":
    st.title("👁️ Preview & Export")
    
    if not st.session_state.get("video_path"):
        st.warning("No video available. Complete the pipeline first.")
        st.stop()
    
    # Video preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.video(st.session_state.video_path)
    
    with col2:
        st.subheader("Dubbed")
        if st.session_state.get("final_video_path") and os.path.exists(st.session_state.final_video_path):
            st.video(st.session_state.final_video_path)
            
            # Audio toggle
            audio_source = st.radio("Audio Source", ["Original", "Dubbed"], horizontal=True)
            if audio_source == "Original":
                st.info("Switch to 'Dubbed' to hear the AI-generated voice")
        else:
            st.info("Final video not ready yet. Complete the pipeline to see results.")
    
    # Stats
    st.divider()
    st.subheader("Project Statistics")
    
    if st.session_state.get("segments"):
        segments = st.session_state.segments
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", len(segments))
        with col2:
            speakers = set(s.get("speaker") for s in segments)
            st.metric("Speakers Detected", len(speakers))
        with col3:
            duration = sum(s.get("duration", 0) for s in segments)
            st.metric("Speech Duration", f"{duration:.1f}s")
        with col4:
            approved = sum(1 for s in segments if s.get("status") == "approved")
            st.metric("Approved", f"{approved}/{len(segments)}")
    
    # Downloads
    st.divider()
    st.subheader("📥 Download Files")
    
    project_dir = st.session_state.get("project_dir", "")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get("final_video_path") and os.path.exists(st.session_state.final_video_path):
            with open(st.session_state.final_video_path, "rb") as f:
                st.download_button(
                    "📹 Download Dubbed Video (MP4)",
                    f,
                    file_name=f"{st.session_state.project_id}_dubbed.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    
    with col2:
        srt_path = os.path.join(project_dir, "subtitles.srt")
        if os.path.exists(srt_path):
            with open(srt_path, "r", encoding="utf-8") as f:
                st.download_button(
                    "📄 Download Subtitles (SRT)",
                    f.read(),
                    file_name=f"{st.session_state.project_id}.srt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    with col3:
        if st.session_state.get("segments"):
            import pandas as pd
            df = pd.DataFrame(st.session_state.segments)
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download Script (CSV)",
                csv,
                file_name=f"{st.session_state.project_id}_script.csv",
                mime="text/csv",
                use_container_width=True
            )


def execute_pipeline():
    """Execute full pipeline with checkpointing"""
    project_dir = st.session_state.project_dir
    video_path = st.session_state.video_path
    
    # Setup logging
    log_path = os.path.join(project_dir, "pipeline_log.txt")
    
    def log(msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(line.strip())
    
    try:
        # Stage 1: Ingest
        st.session_state.pipeline_status["ingest"] = "running"
        st.session_state.pipeline_progress["ingest"] = 0
        log("Stage 1: Ingesting video...")
        
        audio_path, video_info = stage1_ingest(video_path, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"ingest": p}))
        st.session_state.audio_path = audio_path
        st.session_state.video_info = video_info
        st.session_state.pipeline_status["ingest"] = "complete"
        st.session_state.pipeline_progress["ingest"] = 100
        save_checkpoint(project_dir, "stage1", {"audio_path": audio_path, "video_info": video_info})
        log("✅ Stage 1 complete")
        
        # Stage 2: Transcribe
        st.session_state.pipeline_status["transcribe"] = "running"
        log("Stage 2: Transcribing audio...")
        
        model_size = {"fast": "small", "balanced": "medium", "studio": "large-v3"}[st.session_state.quality_mode]
        segments = stage2_transcribe(audio_path, model_size, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"transcribe": p}))
        st.session_state.segments = segments
        st.session_state.pipeline_status["transcribe"] = "complete"
        st.session_state.pipeline_progress["transcribe"] = 100
        save_checkpoint(project_dir, "segments", segments)
        log(f"✅ Stage 2 complete - {len(segments)} segments")
        
        # Stage 3: Diarize
        st.session_state.pipeline_status["diarize"] = "running"
        log("Stage 3: Detecting speakers...")
        
        segments, speakers = stage3_diarize(audio_path, segments, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"diarize": p}))
        st.session_state.segments = segments
        st.session_state.speakers = speakers
        st.session_state.pipeline_status["diarize"] = "complete"
        st.session_state.pipeline_progress["diarize"] = 100
        save_checkpoint(project_dir, "segments", segments)
        save_checkpoint(project_dir, "speakers", speakers)
        log(f"✅ Stage 3 complete - {len(speakers)} speakers")
        
        # Stage 4: Translate
        st.session_state.pipeline_status["translate"] = "running"
        log("Stage 4: Translating with Cerebras Qwen...")
        
        segments = stage4_translate_with_qwen(
            segments, 
            st.session_state.source_lang, 
            st.session_state.target_lang, 
            project_dir,
            cerebras_api_key=st.session_state.get("cerebras_api_key"),
            progress_callback=lambda p: st.session_state.pipeline_progress.update({"translate": p})
        )
        st.session_state.segments = segments
        st.session_state.pipeline_status["translate"] = "complete"
        st.session_state.pipeline_progress["translate"] = 100
        save_checkpoint(project_dir, "segments", segments)
        log(f"✅ Stage 4 complete")
        
        # Stage 5: Emotion Detection
        if st.session_state.enable_emotion:
            st.session_state.pipeline_status["emotion"] = "running"
            log("Stage 5: Detecting emotions...")
            
            segments = stage5_detect_emotions(audio_path, segments, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"emotion": p}))
            st.session_state.segments = segments
            st.session_state.pipeline_status["emotion"] = "complete"
            st.session_state.pipeline_progress["emotion"] = 100
            save_checkpoint(project_dir, "segments", segments)
            log(f"✅ Stage 5 complete")
        else:
            st.session_state.pipeline_status["emotion"] = "complete"
            st.session_state.pipeline_progress["emotion"] = 100
        
        # Stage 6: Clone and Dub (StyleTTS 2)
        st.session_state.pipeline_status["clone_dub"] = "running"
        log("Stage 6: Cloning voices and generating dubs with StyleTTS 2...")
        
        segments = stage6_clone_and_dub_styletts(
            segments, 
            st.session_state.speakers, 
            st.session_state.target_lang,
            project_dir,
            progress_callback=lambda p: st.session_state.pipeline_progress.update({"clone_dub": p})
        )
        st.session_state.segments = segments
        st.session_state.pipeline_status["clone_dub"] = "complete"
        st.session_state.pipeline_progress["clone_dub"] = 100
        save_checkpoint(project_dir, "segments", segments)
        log(f"✅ Stage 6 complete")
        
        # Stage 7: Separate Stems
        st.session_state.pipeline_status["separate"] = "running"
        log("Stage 7: Separating audio stems...")
        
        vocals_path, no_vocals_path = stage7_separate_stems(audio_path, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"separate": p}))
        st.session_state.vocals_path = vocals_path
        st.session_state.no_vocals_path = no_vocals_path
        st.session_state.pipeline_status["separate"] = "complete"
        st.session_state.pipeline_progress["separate"] = 100
        save_checkpoint(project_dir, "stems", {"vocals": vocals_path, "no_vocals": no_vocals_path})
        log(f"✅ Stage 7 complete")
        
        # Stage 8: Align Audio
        st.session_state.pipeline_status["align"] = "running"
        log("Stage 8: Aligning audio timing...")
        
        dubbed_audio_path = stage8_align_audio(segments, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"align": p}))
        st.session_state.dubbed_audio_path = dubbed_audio_path
        st.session_state.pipeline_status["align"] = "complete"
        st.session_state.pipeline_progress["align"] = 100
        save_checkpoint(project_dir, "dubbed_audio", dubbed_audio_path)
        log(f"✅ Stage 8 complete")
        
        # Stage 9: Mix Audio
        st.session_state.pipeline_status["mix"] = "running"
        log("Stage 9: Mixing audio with broadcast standards...")
        
        if st.session_state.keep_bgm:
            mixed_audio_path = stage9_mix_audio(dubbed_audio_path, no_vocals_path, project_dir, progress_callback=lambda p: st.session_state.pipeline_progress.update({"mix": p}))
        else:
            mixed_audio_path = dubbed_audio_path
        
        st.session_state.mixed_audio_path = mixed_audio_path
        st.session_state.pipeline_status["mix"] = "complete"
        st.session_state.pipeline_progress["mix"] = 100
        log(f"✅ Stage 9 complete")
        
        # Stage 10: Lip Sync
        if st.session_state.enable_lipsync:
            st.session_state.pipeline_status["lipsync"] = "running"
            log("Stage 10: Syncing lips...")
            
            lipsynced_video_path = stage10_lipsync(video_path, mixed_audio_path, project_dir, st.session_state.quality_mode, progress_callback=lambda p: st.session_state.pipeline_progress.update({"lipsync": p}))
            st.session_state.lipsynced_video_path = lipsynced_video_path
            st.session_state.pipeline_status["lipsync"] = "complete"
            st.session_state.pipeline_progress["lipsync"] = 100
            log(f"✅ Stage 10 complete")
        else:
            st.session_state.lipsynced_video_path = video_path
            st.session_state.pipeline_status["lipsync"] = "complete"
            st.session_state.pipeline_progress["lipsync"] = 100
        
        # Stage 11: Render
        st.session_state.pipeline_status["render"] = "running"
        log("Stage 11: Rendering final video...")
        
        final_video_path, srt_path = stage11_render(
            st.session_state.lipsynced_video_path,
            mixed_audio_path,
            segments,
            project_dir,
            progress_callback=lambda p: st.session_state.pipeline_progress.update({"render": p})
        )
        st.session_state.final_video_path = final_video_path
        st.session_state.pipeline_status["render"] = "complete"
        st.session_state.pipeline_progress["render"] = 100
        log(f"✅ Stage 11 complete - Final video: {final_video_path}")
        
        log("🎉 PIPELINE COMPLETE!")
        
    except Exception as e:
        import traceback
        error_msg = f"Pipeline failed: {str(e)}"
        st.session_state.error_msg = error_msg
        st.session_state.error_traceback = traceback.format_exc()
        log(f"❌ ERROR: {error_msg}")
        log(traceback.format_exc())
        
        # Mark current stage as failed
        for stage, status in st.session_state.pipeline_status.items():
            if status == "running":
                st.session_state.pipeline_status[stage] = "failed"
                st.session_state[f"error_{stage}"] = str(e)


if __name__ == "__main__":
    pass
