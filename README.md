# DubFlow Studio 🎬

Professional AI-powered video dubbing application with Cerebras Qwen integration and StyleTTS 2 duration control.

## 🚀 Key Improvements Over Original

1. **Cerebras Qwen Translation** - Context-aware, culturally-adapted translations with syllable counting
2. **StyleTTS 2 Duration Control** - No more time-stretching artifacts! Generate at exact target duration
3. **Broadcast Audio Standards** - EBU R128 (-23 LUFS) loudness normalization
4. **Pipeline Checkpointing** - Resume from any stage if interrupted
5. **Professional Audio Mastering** - De-essing, crossfades, BGM ducking
6. **Streamlit UI** - 4-page interface: Upload, Monitor, Editor, Export

## 📁 Project Structure

```
dubflow_studio/
├── app.py                    # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── config.py            # Configuration constants
│   ├── state_manager.py     # Session state management
│   ├── models.py            # Model loading & caching
│   ├── pipeline.py          # All 11 pipeline stages
│   ├── audio.py             # Audio processing utilities
│   └── video.py             # Video processing utilities
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ Setup Instructions

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install FFmpeg**:
```bash
# Windows (with chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

3. **Set Environment Variables**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Download Wav2Lip Model** (optional):
```bash
mkdir -p Wav2Lip/checkpoints
# Download wav2lip_gan.pth from HuggingFace
```

5. **Run the App**:
```bash
streamlit run app.py
```

## 🤖 AI Pipeline

| Stage | Technology | Purpose |
|-------|-----------|---------|
| 1 | FFmpeg | Video/audio extraction |
| 2 | faster-whisper | Transcription |
| 3 | pyannote.audio | Speaker diarization |
| 4 | **Cerebras Qwen** | Translation + script adaptation |
| 5 | wav2vec2 | Emotion detection |
| 6 | **StyleTTS 2** | Duration-controlled TTS |
| 7 | Demucs | Audio separation |
| 8 | pydub | Audio assembly |
| 9 | pyloudnorm | Broadcast mastering |
| 10 | Wav2Lip | Lip synchronization |
| 11 | FFmpeg | Final render |

## 🎯 Key Features

### Duration-Controlled Generation
Unlike the original implementation that used `atempo` time-stretching (causing chipmunk artifacts), StyleTTS 2 generates speech at the exact target duration:

```python
# No stretching needed!
wav = model.inference(
    text=text,
    ref_audio_path=sample_path,
    target_duration=2.5,  # Exact seconds
)
```

### Cerebras Qwen Script Adaptation
The LLM adapts translations to fit timing by counting syllables:

```python
prompt = f"""
Translate with {syllable_target} syllables for {duration}s duration.
Original: {original_text}
"""
```

### Broadcast Audio Standards
- **Loudness**: -23 LUFS (EBU R128)
- **De-essing**: Reduces sibilance
- **Crossfades**: 20ms between segments
- **BGM Ducking**: -12dB when speech present

## 📊 UI Pages

1. **Upload & Settings** - Video upload, language selection, API keys
2. **Pipeline Monitor** - Real-time progress, logs, retry failed stages
3. **Script Editor** - Edit translations, emotions, re-dub segments
4. **Preview & Export** - Compare videos, download outputs

## 🔧 Configuration

Edit `utils/config.py` to customize:
- Sample rates
- Quality modes
- Model selections
- Audio standards

## 📝 API Keys Required

- **HF_TOKEN**: For pyannote.audio speaker diarization (free at huggingface.co)
- **CEREBRAS_API_KEY**: For Qwen translation (get at cerebras.ai)

## 🚀 Future Improvements

- [ ] IP_LAP for better lip-sync
- [ ] Voice embedding persistence across projects
- [ ] Batch processing multiple videos
- [ ] Custom voice library/actor casting

## License

MIT License
