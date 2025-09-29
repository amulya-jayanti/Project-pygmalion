
## System Dependencies

This project requires some system libraries in addition to Python packages.

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv libportaudio2 ffmpeg
```

- `libportaudio2`: Required for `sounddevice` (audio recording/playback)
- `ffmpeg`: Required for `torchaudio` to load/save audio files

### macOS

```bash
brew install portaudio ffmpeg
```

- `portaudio`: Required for `sounddevice`
- `ffmpeg`: Required for `torchaudio`

### Windows

- Download and install [PortAudio binaries](http://www.portaudio.com/download.html) (if needed)
- Download and install [FFmpeg](https://ffmpeg.org/download.html) (add to PATH)

### Python Environment

1. I used venv and the requirements.txt included

# create venv (I called it avatar), then activate it
    python3 -m venv avatar 
    source avatar/bin/activate

# install required dependencies
    pip install -r requirements.txt
    python src/runner.py
    ```

---

**Note:**  
- You must set your `GOOGLE_API_KEY` and `OPENAI_API_KEY` as environment variables or in the code.
- If you encounter audio device errors, ensure your microphone and speakers are connected and accessible.
