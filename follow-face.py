import datetime
import cv2
import numpy as np
import time
import speech_recognition as sr
import sounddevice as sd
import os
import torchaudio
import torch
import threading
from adafruit_servokit import ServoKit
from openai import OpenAI
from TTS.api import TTS

#
debugCameraThread = False;
debugVoiceThread = True;

# shared variable across threads indicating camera finding face
facePresent = False

# ?? common logging method
def log(method, text):
    now = datetime.datetime.now()
    print(str(now) + ": " + method +": " + text)

# Load model ONCE
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False) #gpu requires CUDA/nVidia GPU which is not available on RPi5

# Choose a deep speaker
speaker_id = 'p234'  # Try others too (10=p234?)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#print(os.environ.get("OPENAI_API_KEY"))

# ?? Generate speech waveform
def generate_speech(text):
    log("generate_speech", "Generating...")
    text="In the depths of the void, shadows whisper truths long forgotten.', 'Open your mind to the cosmic dance, let the unseen currents guide you."
    wav = tts.tts(text=text, speaker=speaker_id) #SLOW!!!
    #log("generate_speech", "Returning tensor...") #very fast

    return torch.tensor(wav).unsqueeze(0)

# ?? Apply audio effects (pitch shift, reverb)
def process_audio(wav_tensor, pitch_shift=-3.0): #default to -3.0
    log("process_audio", "Processing audio...")
    # Sample rate from model
    sample_rate = 22050
    
    # Apply pitch shift using torchaudio
    effects = [
        ['pitch', str(int(pitch_shift * 100))],

        ['rate', str(sample_rate)],
        ['reverb', '50'], #works
    ]
    wav_tensor, _ = torchaudio.sox_effects.apply_effects_tensor(wav_tensor, sample_rate, effects)
    return wav_tensor.squeeze().numpy()

# ? Play audio
def play_audio(wav_array):
    log("play_audio", "Play audio...")
    sd.play(wav_array.T, samplerate=22050)
    sd.wait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("? Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return None

def speak(text):
    print(f"?? Creature says: {text}")
    #os.system(f'espeak -ven+m3 -s130 "{text}"')  # Change voice if needed
    os.system(f'tts --text "{text}" --model_name "tts_models/en/vctk/vits" --speaker_idx p234 --out_path raw.wav')
    os.system('sox raw.wav final.wav pitch -400 reverb tempo 0.80')
    os.system('aplay final.wav')

def ask_gpt(prompt):
#    messages = [{"role": "system", "content": "You are an ancient tentacled monster from another dimension. Speak with eerie, cryptic wisdom. Be unsettling but not violent."},
#                {"role": "user", "content": prompt}]
    messages = [{"role": "system", "content": "You are an ancient tentacled monster from another dimension. Speak with eerie, cryptic wisdom. Be unsettling but not violent. Limit your response to one or two sentences."},
                {"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # Or "gpt-3.5-turbo" "gpt-4"
    messages=messages)
    return response.choices[0].message.content

##### this might need to go into its own thread so that the "following" movement
# is not competing with the listening and responding
'''
while True:
    user_input = listen()
    if user_input:
        print(f"Visitor said: {user_input}")
        response = ask_gpt(user_input)
        speak(response)
    else:
        print("Didn't hear anything.")
'''
#####


def speakingTask():
    #initialize variables...
    global facePresent
    global debugVoiceThread
    
    #loop infinitely...
    while True:
        if True: #facePresent:
            if debugVoiceThread: log("speakingTask","face present, listening...")
            user_input = listen()
            if user_input:
                if debugVoiceThread: print(f"Visitor said: {user_input}")
                response = ask_gpt(user_input)
                #speak(response)
                raw_wav = generate_speech(response)
                processed_wav = process_audio(raw_wav, pitch_shift=-3.0)
                play_audio(processed_wav)
            else:
                if debugVoiceThread: print("Didn't hear anything.")
        else:
            if debugVoiceThread: print(f"no face present, make some noise?...")

        if debugVoiceThread: print(f"speakingTask finishing")
        time.sleep(2)
    
thread1 = threading.Thread(target=speakingTask)
thread1.start()

def cameraEyeMovementTask():
    global facePresent
    global debugCameraThread
    
    # Set up the kit for 16-channel PCA9685
    kit = ServoKit(channels=16)

    # Initialize servo angles (you can adjust to center positions)
    pan_channel = 0
    tilt_channel = 1
    pan_angle = 10
    tilt_angle = 10
    kit.servo[pan_channel].angle = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle
    time.sleep(1.0);
    pan_angle = 90
    tilt_angle = 90
    kit.servo[pan_channel].angle = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle
    time.sleep(1.0);

    # Load Haar Cascade for face detection
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Open the USB camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default USB cam


    #loop infinitely...
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            

        # Convert to grayscale (Haar works on gray images)
        if debugCameraThread: print("converting to grayscale.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        if debugCameraThread: print("detecting faces")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around all faces
        '''
        print("draw")
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
'''

        # Initialize variables
        largest_face = None
        max_area = 0

        # Find the largest face
        if debugCameraThread: print("finding largest face")
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)

        # If a face is found, process it
        if largest_face is not None:
            
            #indicate face is present so other threads know
            facePresent = True
            
            if debugCameraThread: print("face found = processing it...")
            x, y, w, h = largest_face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Draw rectangle and center point
            if debugCameraThread: print("draw rectangle around and circle center")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

            # Show the frame
            if debugCameraThread: print("showing image")
            cv2.imshow('Face Tracking', frame)

            # Calculate frame center
            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            # Show offset (can use this to move servos later)
            offset_x = face_center_x - frame_center_x
            offset_y = face_center_y - frame_center_y

            if debugCameraThread: print(f"Face offset: X={offset_x}, Y={offset_y}")

            # Sensitivity (tune this!)
            sensitivity = 0.1  # smaller is more sensitive
            max_angle_change = 2  # limit how fast it can move

            # Normalize offset (based on frame size)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            norm_x = offset_x / frame_width
            norm_y = offset_y / frame_height

            # Compute angle adjustments
            delta_pan = -norm_x * 90 * sensitivity
            delta_tilt = -norm_y * 90 * sensitivity

            # Apply limits
            delta_pan = max(-max_angle_change, min(max_angle_change, delta_pan))
            delta_tilt = max(-max_angle_change, min(max_angle_change, delta_tilt))

            # Update angles
            pan_angle += delta_pan
            tilt_angle += delta_tilt

            # Clamp angles to servo range
            pan_angle = max(0, min(180, pan_angle))
            tilt_angle = max(0, min(180, tilt_angle))

            # Move servos
            if debugCameraThread: print("pan_angle: " + str(pan_angle) + " tilt_angle: " + str(tilt_angle))
            kit.servo[pan_channel].angle = pan_angle
            kit.servo[tilt_channel].angle = tilt_angle
            time.sleep(0.01)
        else:
            facePresent = False
            #return to 

        # Exit on 'q'
        if debugCameraThread: print("check for q")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
cameraThread = threading.Thread(target=cameraEyeMovementTask)
#cameraThread.start()
