# audio_utils.py
from gtts import gTTS
import base64
import os
from datetime import datetime

def generate_podcast_script(summary: str, classification: dict) -> str:
    """Generate formatted podcast script from analysis results"""
    return f"""
    Research Paper Analysis Podcast - {datetime.now().strftime('%Y-%m-%d')}
    
    Here's the summary of the research paper:
    {summary}
    
    Classification Results:
    - Primary Category: {classification.get('category', 'N/A')}
    - Confidence Score: {classification.get('confidence', 0):.1%}
    - Key Keywords: {', '.join(classification.get('keywords', []))}
    
    This analysis was generated using advanced AI models.
    """

def text_to_speech(text: str, filename: str = "podcast.mp3") -> str:
    """Convert text to audio file using gTTS"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        return filename
    except Exception as e:
        raise RuntimeError(f"Audio generation failed: {str(e)}")

def autoplay_audio(file_path: str):
    """Auto-play audio in Streamlit (use with caution)"""
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md

def cleanup_audio_files(file_path: str):
    """Remove temporary audio files"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass