"""Session State Manager for DubFlow Studio"""

import streamlit as st

class StateManager:
    """Manages Streamlit session state initialization and updates"""
    
    @staticmethod
    def init_session_state():
        """Initialize all session state variables"""
        
        defaults = {
            # Project info
            "project_id": None,
            "project_dir": None,
            "video_path": None,
            
            # Settings
            "source_lang": "en",
            "target_lang": "hi",
            "quality_mode": "balanced",
            "use_cerebras": True,
            "cerebras_api_key": "",
            "enable_lipsync": True,
            "keep_bgm": True,
            "enable_emotion": True,
            
            # Pipeline state
            "pipeline_status": {},
            "pipeline_progress": {},
            "pipeline_thread": None,
            "current_stage": None,
            
            # Data
            "segments": [],
            "speakers": {},
            "audio_path": None,
            "video_info": {},
            "vocals_path": None,
            "no_vocals_path": None,
            "dubbed_audio_path": None,
            "mixed_audio_path": None,
            "lipsynced_video_path": None,
            "final_video_path": None,
            
            # Voice embeddings for consistency
            "voice_embeddings": {},
            
            # Editor state
            "redub_queue": [],
            "selected_segment": None,
            
            # Errors
            "error_msg": None,
            "error_traceback": None,
            
            # Logs
            "logs": [],
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_pipeline():
        """Reset pipeline state for new run"""
        st.session_state.pipeline_status = {}
        st.session_state.pipeline_progress = {}
        st.session_state.current_stage = None
        st.session_state.error_msg = None
        st.session_state.error_traceback = None
    
    @staticmethod
    def update_stage_status(stage_name, status, progress=None, error=None):
        """Update stage status in session state"""
        st.session_state.pipeline_status[stage_name] = status
        if progress is not None:
            st.session_state.pipeline_progress[stage_name] = progress
        if error:
            st.session_state[f"error_{stage_name}"] = error
    
    @staticmethod
    def get_stage_status(stage_name):
        """Get current status of a stage"""
        return st.session_state.pipeline_status.get(stage_name, "pending")
