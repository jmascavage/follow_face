import requests
import soundfile as sf
import sounddevice as sd
import io

TTS_SERVER = "http://10.0.0.48:5002"  # Replace with your PC's IP address

def speak(text):
    print(f"Sending to TTS server: {text}")
    response = requests.post(f"{TTS_SERVER}/speak", json={"text": text})
    if response.status_code == 200:
        wav_io = io.BytesIO(response.content)
        data, samplerate = sf.read(wav_io)
        sd.play(data, samplerate)
        sd.wait()
    else:
        print(f"Error: {response.status_code}, {response.text}")

# ? Example
speak("The stars are not aligned, but you are brave to come here.")
