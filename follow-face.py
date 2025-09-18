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
import requests
import soundfile as sf
import sounddevice as sd
import io
from adafruit_servokit import ServoKit
from openai import OpenAI
from TTS.api import TTS
from gpiozero import LED

######################################################
# GLOBAL SETTINGS AND CONSTANTS
######################################################
debugCameraThread = False;
debugVoiceThread = False;
# shared variable across threads indicating camera finding face
facePresent = False
# URL of TTS server
TTS_SERVER = "http://10.0.0.114:5002"  # Replace with IP address where TTS server is running
#ChatGPT base prompt
messages_base = [{"role": "system", "content": "You are an ancient creature from another dimension.  Your name is Faydekin. You are a guardian of space-time - this is your job and what you do. You are wearing a black outfit with silver armor and you have a staff in your right hand with an artifact at the top that looks like a blue diamond inside a clear globe. Speak with eerie, cryptic wisdom using language a twelve year old can understand. Be unsettling but not violent. Limit your response to two or three sentences.  Ensure your response relates to the user's prompt.  If the user says hello you should respond with something including a hello back.  If they ask for a joke or riddle, tell them one. If they ask about candy, tell them that candy can be found at the door to the right, but beware of dangers and stay on the path.  It is Halloween night and you are part of a haunted house.  The haunted house has vampires, scarecrows, ghosts, banshees, and other monsters.  You can share that knowledge in your responses but do not list all of them at once unless asked."}]
messages = []
#define GPIO pin for triggering the solenoid - then can use solenoid.on() and solenoid.off()
solenoid = LED(17, active_high=True)  # Use the GPIO pin number (e.g., 17 for the example)
#start with solenoid in off position (valve should be set up for off=up position)
solenoid.off()
# Load model ONCE
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False) #gpu requires CUDA/nVidia GPU which is not available on RPi5
# Choose a deep speaker
speaker_id = 'p234'  # Try others too (10=p234?)
#get ChatGPT api key from ENV variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#print(os.environ.get("OPENAI_API_KEY"))
defaultPanAngle = 90
defaultTiltAngle = 100
# SERVO MOTION SETTINGS
# 'max_angle_change' caps how far a single angle jump in servo motion can be.  Generally following a
#	a face does not need movements more than 2 degrees at a time, and setting this helps to avoid
#	unexpectedly jerky motion
max_angle_change = 5 # limit how fast it can move - originally 
# "sensitivity" reflects how aggressively to adjust the servo - think of it as a dampener.
# 	Without dampening, the movement will be jittery because the servos and the calculations are not
#   perfect - there will always be an adjustment to make and the servos therefore always moving.
#   The 'sensitivity' is multiplied by the angle of change, and by being a value <1.0, forces changes
#   to be smaller - and no change at all when the change values get low enough (so once the camera is
#   close to centering the face it stops trying to adjust more - thus no jitter).
#   This dampening doees force more steps in getting to a position, but the adjutment loop is fast
sensitivity = .08  # should always be <=1.0. .2 was crazy jittery. .08 seems decent
min_pan_angle = 20		#right shoulder - 20 seems good
max_pan_angle = 160		#left shoulder - 160 seems good
min_tilt_angle = 20 	#toward chest - 20 seems good
max_tilt_angle = 145	#toward sky - 135 seems good

# THIS IS A TESTING LOOP ONLY - used for testing the pneumatics
# Should be set to False for normal operations
while(False):
    
    time.sleep(10.0);
    print("solenoid on...")
    solenoid.on()
    time.sleep(10)
    print("solenoid off...")
    solenoid.off()
    time.sleep(10)
    print("solenoid on...")
    solenoid.on()
    time.sleep(10)
    print("solenoid off...")
    solenoid.off()

# ?? common logging method
def log(method, text):
    now = datetime.datetime.now()
    print(str(now) + ": " + method +": " + text)

bodyRaisedFlag=True

def raiseBody():
    global bodyRaisedFlag
    global solenoid
    solenoid.off()
    bodyRaisedFlag=True
    
def isBodyRaised():
    global bodyRaisedFlag
    return bodyRaisedFlag
    #global solenoid
    #return solenoid.is_lit

def lowerBody():
    global bodyRaisedFlag
    global solenoid
    solenoid.on()
    bodyRaisedFlag=False
    
# ?? Generate speech waveform
def generate_speech(text):
    log("generate_speech", "Generating...")
#    text="In the depths of the void, shadows whisper truths long forgotten.', 'Open your mind to the cosmic dance, let the unseen currents guide you."
    wav = tts.tts(text=text, speaker=speaker_id) #SLOW!!!
    #log("generate_speech", "Returning tensor...") #very fast

    return torch.tensor(wav).unsqueeze(0)

# ?? Generate speech waveform
def generate_speech_tts_server(text):
    log("generate_speech_tts_server", "Generating:" + text)
    response = requests.post(f"{TTS_SERVER}/speak", json={"text": text})
    
    if response.status_code == 200:
        wav_io = io.BytesIO(response.content)
        data, samplerate = sf.read(wav_io, dtype='float32')  # Ensure float32 for torch
        tensor = torch.tensor(data).unsqueeze(0)  # Add batch dimension: [1, samples]
#        return tensor, samplerate
        return tensor
    else:
        print(f"Error: {response.status_code}, {response.text}")

    return


# ?? Apply audio effects (pitch shift, reverb)
def process_audio(wav_tensor, pitch_shift=-3.0): #default to -3.0
    log("process_audio", "Processing audio...")
    # Sample rate from model
    sample_rate = 22050
    
    # Apply pitch shift using torchaudio
    effects = [
        ['pitch', str(int(pitch_shift * 100))],
        ['speed', '0.95'], #must be before 'rate' effect in list

        ['rate', str(sample_rate)],
        ['reverb', '20'], #higher numbers gives more reverb - default is 50
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
        log("listen", "? Listening...")
        audio = r.adjust_for_ambient_noise(source)
#        audio = r.listen(source,None,None)
        audio = r.listen(source,timeout=None, phrase_time_limit=None) #still times out after 10s
        
#        audio = r.listen(source)
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

def reset_gpt_session():
    global messages
    messages = messages_base.copy();
    
reset_gpt_session()

def get_gpt_messages():
    return messages
    
def ask_gpt(prompt):
    messages = get_gpt_messages()
#    messages = [{"role": "system", "content": "You are an ancient tentacled monster from another dimension. Speak with eerie, cryptic wisdom. Be unsettling but not violent."},
#                {"role": "user", "content": prompt}]
#    messages = [{"role": "system", "content": "You are an ancient creature from another dimension.  Your name is Faydekin. You are a guardian of space-time - this is your job and what you do. Speak with eerie, cryptic wisdom. Be unsettling but not violent. Limit your response to two or three sentences.  Ensure your response relates to the user's prompt.  If the user says hello you should respond with something including a hello back.  If they ask for a joke or riddle, tell them one. If they ask about candy, tell them that candy can be found at the door to the right, but beware of dangers and stay on the path."},
#                {"role": "user", "content": prompt}]
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # Or "gpt-3.5-turbo" "gpt-4"
    messages=messages)
    log("ask_gpt","chatgpt response: " + response.choices[0].message.content)
    messages.append({"role": "assistant", "content": response.choices[0].message.content})    #adds ChatGPT response to "session" context across calls 9999
    return response.choices[0].message.content

##### running listen/speaking in its own thread so it is not competing with visual following
#     keeps code a little cleaner as well
#     there is a global variable "facePresent" dependency with the visual following task, which is
#     intended to allow the speakingTask to know if a face is being followed (and behave differently
#     when not - e.g., reset the accumulation of session data with ChatGPT) 
#####

def speakingTask():
    #link global variables...
    global facePresent
    global debugVoiceThread
    
    if debugVoiceThread: log("speakingTask", "Entering function...")
        
    #loop infinitely...
    while True:
        try:
            if facePresent:
                if debugVoiceThread: log("speakingTask","face present, listening...")
                user_input = listen()
                if not user_input:
                    if debugVoiceThread: log("speakingTask","face present, listening2...")
                    user_input = listen()
                if not user_input:
                    if debugVoiceThread: log("speakingTask","face present, listening3...")
                    user_input = listen()
                    
                if user_input:
                    if debugVoiceThread: log("speakingTask", f"Visitor said: {user_input}")
                    response = ask_gpt(user_input)
                    #speak(response)
    #                raw_wav = generate_speech(response)
                    raw_wav = generate_speech_tts_server(response)
                    processed_wav = process_audio(raw_wav, pitch_shift=-3.0)
                    play_audio(processed_wav)
                else:
                    if debugVoiceThread: log("speakingTask", "Didn't hear anything...")
                    response = ask_gpt("You have a person standing before you but is not saying anything.  Speak something to them, such as 'Why do you stand there in silence.  Speak wanderer, and I will listen.'")
                    #response = "Why do you stand there in silence.  Speak wanderer, and I will listen."
                    raw_wav = generate_speech_tts_server(response)
                    processed_wav = process_audio(raw_wav, pitch_shift=-3.0)
                    play_audio(processed_wav)          
            else:
                if debugVoiceThread: log("speakingTask", "no face present, make some noise?...")

            if debugVoiceThread: log("speakingTask", "speakingTask loop finishing/repeating")

        except Exception as e:
            # Catches any other unexpected exceptions
            print(f"An unexpected error occurred: {e}")
    
    print("ENDING SPEAKING LOOP!!!")
    
speakingThread = threading.Thread(target=speakingTask)
speakingThread.start()

def cameraEyeMovementTask():
    global facePresent
    global defaultPanAngle
    global defaultTiltAngle
    global debugCameraThread
    global sensitivity
    global max_angle_change
    global min_pan_angle
    global max_pan_angle
    global min_tilt_angle
    global max_tilt_angle

    timeLastFacePresent = datetime.datetime.now()

    if debugCameraThread: log("cameraEyeMovementTask", "Entering function...")

    # Set up the kit for 16-channel PCA9685
    kit = ServoKit(channels=16)

    # Initialize servo angles (you can adjust to center positions)
    pan_channel = 0
    tilt_channel = 1
    log("cameraEyeMovementTask", "Start by panning/tilting to defaults...")
    pan_angle = defaultPanAngle
    tilt_angle = defaultTiltAngle
    kit.servo[pan_channel].angle = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle
    time.sleep(.01);
        
    # Load Haar Cascade for face detection
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Open the USB camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default USB cam

    if debugCameraThread: log("cameraEyeMovementTask", "Beginning infinite loop...")
    #loop infinitely...
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
                

            # Convert to grayscale (Haar works on gray images)
            #if debugCameraThread: log("cameraEyeMovementTask", "converting to grayscale.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            #if debugCameraThread: log("cameraEyeMovementTask", "detecting faces")
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
            #if debugCameraThread: log("cameraEyeMovementTask", "finding largest face")
            for (x, y, w, h) in faces:
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face = (x, y, w, h)

            # Show the frame - this will show even with no face present (helps debug and focus camera)
            #if debugCameraThread: log("cameraEyeMovementTask", "showing image")
            cv2.imshow('Face Tracking', frame)
                
            # If a face is found, process it
            if largest_face is not None:
                
                #indicate face is present so other threads know
                facePresent = True
                timeLastFacePresent = datetime.datetime.now()
                
                #if debugCameraThread: log("cameraEyeMovementTask", "face found = processing it...")
                x, y, w, h = largest_face
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Draw rectangle and center point
                #if debugCameraThread: log("cameraEyeMovementTask", "draw rectangle around and circle center")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

                # Show the frame
                #if debugCameraThread: log("cameraEyeMovementTask", "showing image")
                cv2.imshow('Face Tracking', frame)

                # Calculate frame center
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Calculate offset of center of face from center of frame (can use this to move servos later)
                offset_x = face_center_x - frame_center_x
                offset_y = face_center_y - frame_center_y

                #if debugCameraThread: log("cameraEyeMovementTask", f"Face offset: X={offset_x}, Y={offset_y}")


                # Normalize offset (based on frame size)
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                #if debugCameraThread: log("cameraEyeMovementTask", f"Frame size: Width(max X)={frame_width}, Height(max Y)={frame_height}")

                norm_x = offset_x / frame_width # norm_x = the % of frame width to center of face
                norm_y = offset_y / frame_height

                # Compute angle adjustments - using 'sensitivity' as a dampener
                # norm_x and norm_y are the percent across the frame needed to move to center face
                # if we multiply that by that percent by the field of view angle (number of degrees in
                # a frame seen by the camera) that tells us how many degrees to move
                # the 'sensitivity' helps avoid jitter - the calculation and motion isn't perfect so if
                # we try to always jump exactly we get a jitter movement, the sensitivity knocks small
                # changes down to no change
                delta_pan = -norm_x * 120 * sensitivity
                delta_tilt = -norm_y * 120 * sensitivity
                #if debugCameraThread: log("cameraEyeMovementTask", f"Deltas before limits: delta_pan={delta_pan}, delta_tilt={delta_tilt}")

                # Do not allow jumps greater than max_angle_change
                delta_pan = max(-max_angle_change, min(max_angle_change, delta_pan))
                delta_tilt = max(-max_angle_change, min(max_angle_change, delta_tilt))

                # Update angles
                pan_angle += delta_pan
                tilt_angle += delta_tilt

                # Do not allow angles to exceed desired servo range

                pan_angle = max(min_pan_angle, min(max_pan_angle, pan_angle))
                tilt_angle = max(min_tilt_angle, min(max_tilt_angle, tilt_angle))

                # Move servos
                if debugCameraThread: log("cameraEyeMovementTask", "pan_angle: " + str(pan_angle) + " tilt_angle: " + str(tilt_angle))
                kit.servo[pan_channel].angle = pan_angle
                kit.servo[tilt_channel].angle = tilt_angle
                
                log("cameraEyeMovementTask", "Checking Raise/Lower: Raised? " + str(isBodyRaised()) + " tilt_angle: " + str(tilt_angle))
                if(isBodyRaised() and tilt_angle < 92):
                    tilt_angle += 15
                    log("cameraEyeMovementTask", "Lowering body... tilt_angle going to: " + str(tilt_angle))
                    kit.servo[tilt_channel].angle = tilt_angle
                    time.sleep(0.0)
                    log("cameraEyeMovementTask", "completed servo call...")
                    lowerBody()
                    log("cameraEyeMovementTask", "completed lowerBody() call...")
                if(not isBodyRaised() and tilt_angle >125):
                    log("cameraEyeMovementTask", "Raising body... pan_angle going to: " + str(pan_angle))
                    raiseBody()
                time.sleep(0.001) #time between adjustments - .01 makes movement smooth, high values can make movement jumpy
            else:
                timeSinceLastFacePresent = (datetime.datetime.now() - timeLastFacePresent).total_seconds() * 1000
                if debugCameraThread: log("cameraEyeMovementTask", "Lost track of face for " + str(timeSinceLastFacePresent) + "ms")
                
                if timeSinceLastFacePresent > 10000:
                    facePresent = False
                    reset_gpt_session()
                    pan_angle = defaultPanAngle
                    tilt_angle = defaultTiltAngle
                    raiseBody()
                    kit.servo[pan_channel].angle = pan_angle
                    kit.servo[tilt_channel].angle = tilt_angle
                    time.sleep(2.00)                

            # Exit on 'q'
            if debugCameraThread: log("cameraEyeMovementTask", "check for q")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            # Catches any other unexpected exceptions
            print(f"An unexpected error occurred: {e}")
            
    print("ENDING CAMERA LOOP!!!")
    cap.release()
    cv2.destroyAllWindows()
    
cameraThread = threading.Thread(target=cameraEyeMovementTask)
cameraThread.start()
