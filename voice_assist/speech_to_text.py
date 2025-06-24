import os
import queue
import sounddevice as sd
import vosk
import json

# Path to the Vosk model
MODEL_PATH = "vosk-model-en-in-0.5"

# Global variable to control stopping
stop_listening = False

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at '{MODEL_PATH}'. Download from https://alphacephei.com/vosk/models")

model = vosk.Model(MODEL_PATH)
sample_rate = 16000
q = queue.Queue()

def callback(indata, frames, time, status):
    """Callback function to put audio data into a queue."""
    if status:
        print(f"⚠️ Audio status: {status}", flush=True)
    q.put(bytes(indata))

def speech_to_text():
    """Captures audio and converts it to text using Vosk (offline)."""
    global stop_listening
    stop_listening = False

    print("\n🎤 Listening (Vosk)...", flush=True)

    try:
        with sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            rec = vosk.KaldiRecognizer(model, sample_rate)

            while True:
                if stop_listening:
                    return ""

                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        return text
                # Optionally, handle partial results
                # else:
                    # print("Cannot recognize speech yet...", flush=True)
    except Exception as e:
        print(f"\nAn error occurred: {e}", flush=True)
        return ""

def stop_speech_recognition():
    """Sets the global flag to stop recognition."""
    global stop_listening
    stop_listening = True
    print("🛑 Speech recognition stopped by user", flush=True)

# Test Run
if __name__ == "__main__":
    print("Transcribed Text:", speech_to_text())
