import os
import sounddevice as sd
import torch
import torchaudio
import openai
import google.generativeai as genai
import dotenv
import google.api_core.exceptions as gexc
import soundfile as sf
import numpy as np
dotenv.load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# Try to use local whisper for transcription
try:
    import whisper
    _whisper_model = None
except Exception:
    whisper = None
    _whisper_model = None

# Try to use local pyttsx3 for TTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# 🎤 Step 1: Record audio
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    # Diagnostics
    print("🎤 Recording... speak now!")
    try:
        print("Default device:", sd.default.device)
        print("All devices:")
        for i, d in enumerate(sd.query_devices()):
            print(i, d["name"], "max_input:", d.get("max_input_channels"))
    except Exception as e:
        print("Could not query devices:", e)

    # Record
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Recorded shape:", audio.shape, "dtype:", audio.dtype)
    print("Recorded level: min=%g max=%g mean=%g" % (float(audio.min()), float(audio.max()), float(np.abs(audio).mean())))
    if abs(audio).max() < 1e-5:
        print("⚠️  Recorded signal is near zero. Check microphone, input device, and volume/mute settings.")

    # Try torchaudio if backend exists, otherwise use soundfile
    backends = []
    try:
        backends = torchaudio.list_audio_backends()
    except Exception:
        backends = []

    try:
        if backends:
            waveform = torch.from_numpy(audio).squeeze()
            # torchaudio expects (channels, frames)
            torchaudio.save(filename, waveform.unsqueeze(0), samplerate, format="wav")
        else:
            # soundfile expects (frames, channels) or 1D for mono
            sf.write(filename, np.squeeze(audio), samplerate)
    except Exception as e:
        print("Failed to save with preferred backend:", e)
        # final attempt with soundfile
        try:
            sf.write(filename, np.squeeze(audio), samplerate)
        except Exception as e2:
            raise RuntimeError("Failed to save audio file: " + str(e2)) from e2

    print("✅ Saved:", filename)
    return filename

# 📝 Step 2: Transcribe with local Whisper (no OpenAI API)
def transcribe(filename):
    if whisper is None:
        raise RuntimeError(
            "Local transcription requires the 'openai-whisper' package. Install with: pip install -U openai-whisper"
        )
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (this may take a while)...")
        _whisper_model = whisper.load_model("base")  # change model size as needed
    result = _whisper_model.transcribe(filename)
    text = result.get("text", "").strip()
    print("🗣️ You said:", text)
    return text

# 🤖 Step 3: Send to Gemini LLM (unchanged)
def chat_with_gemini(prompt):
    """
    Send prompt to configured Google model.
    Set GOOGLE_MODEL in your environment to a model that supports generate_content.
    Example: export GOOGLE_MODEL="models/text-bison-001"
    """
    model_name = os.environ.get("GOOGLE_MODEL", "models/text-bison-001")
    try:
        response = genai.GenerativeModel(model_name).generate_content(prompt)
        text = response.text
        print("🤖 Gemini:", text)
        return text
    except gexc.NotFound as e:
        # Helpful error for the user
        msg = (
            f"Model '{model_name}' not found or not supported for generate_content (404).\n"
            "Check available models in the Google Cloud Console -> Generative AI -> Models, "
            "or set GOOGLE_MODEL to a supported model name.\n"
            "Example supported model: models/text-bison-001\n"
            "If unsure, visit: https://cloud.google.com/vertex-ai/docs/generative-ai/overview\n"
            f"\nOriginal error: {e}"
        )
        raise RuntimeError(msg) from e
    except Exception:
        raise

# 🔊 Step 4: Text-to-Speech using local pyttsx3 (no OpenAI API)
def speak(text, filename="output.wav"):
    if pyttsx3 is None:
        raise RuntimeError(
            "Local TTS requires the 'pyttsx3' package. Install with: pip install pyttsx3"
        )
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    print("🔊 Saved speech:", filename)
    # playback
    try:
        sd.play(torchaudio.load(filename)[0].T.numpy(), 24000)
        sd.wait()
    except Exception:
        print("Playback failed; file saved at", filename)

# 🔁 Main loop
if __name__ == "__main__":
    audio_file = record_audio()
    user_text = transcribe(audio_file)
    llm_reply = chat_with_gemini(user_text)
    speak(llm_reply)
