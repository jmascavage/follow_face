#!/usr/bin/env python3
"""
follow-face.py  (latency-optimized)

Pipeline per visitor utterance:
    listen -> GPT (streamed) -> sentence queue -> TTS server -> audio queue -> player
Each stage runs in its own thread, so sentence 1 is being spoken while sentence 2 is still
being generated / synthesized.  A "Translating..." filler is played only if the first real
audio has not been queued within FILLER_DELAY seconds.
"""
import audioop                  # removed in Python 3.13 - if the import fails: pip install audioop-lts
import datetime
import io
import os
import queue
import random
import re
import threading
import time

import cv2
import numpy as np
import requests
import soundfile as sf
import sounddevice as sd
import speech_recognition as sr
from adafruit_servokit import ServoKit
from gpiozero import LED
from openai import OpenAI

######################################################
# GLOBAL SETTINGS AND CONSTANTS
######################################################
debugCameraThread = True
debugVoiceThread = False
PNEUMATICS_TEST = False      # True = run the solenoid test loop forever instead of the app
SHOW_PREVIEW = False          # False = no cv2 window (a little faster, no focus/debug view)

# shared variable across threads indicating camera finding face
facePresent = False
# shared variable: True while the mic is open waiting for the visitor to speak.  The camera thread
# reads it to hold the servos still (servo noise is as loud as speech at the mic).
listening = False

# --- TTS server ---
TTS_SERVER = "http://10.0.0.168:5002"  # Replace with IP address where TTS server is running
TTS_SPEAKER = "p234"
TTS_TIMEOUT = 30
PITCH_SHIFT = -1.0           # semitones; applied on the server if pedalboard is installed there,
                             # otherwise locally with sox (the original behavior)

# --- GPT ---
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 120
MAX_HISTORY = 12             # user+assistant messages kept after the system prompt (keep even)
MIN_CHARS = 15               # merge very short sentences so TTS chunks don't get choppy

# --- Speech recognition ---
LISTEN_TIMEOUT = 8           # seconds of silence before the "why do you stand there" line
PHRASE_LIMIT = 12            # max seconds for a single utterance
PAUSE_THRESHOLD = 0.9        # silence that ends an utterance (library default 0.8) - long enough
                             # that a visitor pausing to think isn't cut into fragments
MIN_PHRASE_SECONDS = 0.35    # audio shorter than this is treated as noise and ignored
REPEAT_MIN_SECONDS = 1.0     # only say a "repeat that" line if the audio we couldn't understand was
                             # at least this long; anything shorter is treated as noise and ignored
ENERGY_MARGIN = 1.0          # multiply the calibrated noise level by this (higher = less sensitive)
MIN_ENERGY = 1200             # speech threshold is never lower than this (stops over-sensitivity)
MAX_ENERGY = 1500            # ...and never higher than this (stops it going deaf if calibration
                             #    happened to run during servo/pump noise)
DYNAMIC_THRESHOLD = False     # threshold keeps adapting to room noise
MIC_TEST = False             # True = just print live mic levels vs. threshold, to tune the above

# --- Keep the servos quiet while the visitor is talking ---
# Offsets are a fraction of the frame size (0.10 = face 10% of the frame width/height off center).
# While the mic is listening, the head only moves if the face is more than LISTEN_DEADBAND off
# center, so a visitor standing and talking gets a silent creature.  When the creature is speaking
# or thinking (mic closed) it re-centers on the face normally.  0.0 disables the whole feature.
LISTEN_DEADBAND = 0.12
IDLE_DEADBAND = 0.02         # tiny dead zone the rest of the time - stops constant micro-buzzing
POST_SPEECH_PAUSE = 0.15     # extra pause before the mic reopens
ECHO_TAIL = 0.4              # seconds of quiet after the creature stops, before listening again
                             # (raise if the reverb/room echo is still being picked up)

# --- Behavior ---
FILLER_DELAY = 0.7           # play "Translating..." only if no speech audio is queued by then
                             # (0.0 = always play it immediately, in parallel with GPT)
GREET_ON_ARRIVAL = False     # True = speak a canned greeting the moment a new face appears

# --- Camera ---
DETECT_WIDTH = 320           # run face detection on a downscaled frame
FACE_LOST_MS = 10000         # no face for this long = visitor left, reset the session

# ChatGPT base prompt
messages_base = [{"role": "system", "content": "You are an ancient creature from another dimension.  Your name is Faydekin. You are a guardian of space-time - this is your job and what you do. You are wearing a black outfit with silver armor and you have a staff in your right hand with an artifact at the top that looks like a blue diamond inside a clear globe. Speak with eerie, cryptic wisdom using language a twelve year old can understand. Be unsettling but not violent. Limit your response to two or three sentences.  Ensure your response relates to the user's prompt.  If the user says hello you should respond with something including a hello back.  If they ask for a joke or riddle, tell them one. If they ask about candy, tell them that candy can be found at the door down the dark corridor to the left, but beware of dangers and stay on the path.  It is Halloween night and you are part of a haunted house.  The haunted house has vampires, scarecrows, ghosts, banshees, witches, and other monsters.  You can share that knowledge in your responses but do not list all of them at once unless asked."}]
messages = []
session_id = 0
history_lock = threading.Lock()

# Canned lines - pre-rendered once at startup so they cost zero latency during a visit
SILENCE_LINES = [
    "Why do you stand there in silence? Speak, wanderer, and I will listen.",
    "The void grows restless with your silence. Speak, traveler.",
    "I sense you, small one. Do not be afraid. Ask, and the stars may answer.",
    "Even the ghosts here are whispering. Will you not speak?",
]
REPEAT_LINES = [
    "Speak now"]
#    "Your words were lost in the void. Speak again, wanderer.",
#    "The shadows swallowed your voice. Say it once more.",
#]
GREETING_LINES = [
    "Ah. A visitor drifts into my domain. Welcome, wanderer.",
    "Time bends around you, traveler. Come closer, and speak.",
]
FILLER_TEXT = "Translating..."
FILLER_CLIP = None

# define GPIO pin for triggering the solenoid - then can use solenoid.on() and solenoid.off()
solenoid = LED(17, active_high=True)
# start with solenoid in off position (valve should be set up for off=up position)
solenoid.off()

# get ChatGPT api key from ENV variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

defaultPanAngle = 90
defaultTiltAngle = 110

# SERVO MOTION SETTINGS
# 'max_angle_change' caps how far a single angle jump in servo motion can be.  Generally following a
#   face does not need movements more than 2 degrees at a time, and setting this helps to avoid
#   unexpectedly jerky motion
max_angle_change = 5
# "sensitivity" reflects how aggressively to adjust the servo - think of it as a dampener.
#   Without dampening, the movement will be jittery because the servos and the calculations are not
#   perfect - there will always be an adjustment to make and the servos therefore always moving.
#   The 'sensitivity' is multiplied by the angle of change, and by being a value <1.0, forces changes
#   to be smaller - and no change at all when the change values get low enough (so once the camera is
#   close to centering the face it stops trying to adjust more - thus no jitter).
sensitivity = .08  # should always be <=1.0. .2 was crazy jittery. .08 seems decent
min_pan_angle = 20      # right shoulder - 20 seems good
max_pan_angle = 160     # left shoulder - 160 seems good
min_tilt_angle = 20     # toward chest - 20 seems good
max_tilt_angle = 145    # toward sky


def log(method, text):
    now = datetime.datetime.now()
    print(str(now) + ": " + method + ": " + text)


######################################################
# PNEUMATICS
######################################################
bodyRaisedFlag = True


def raiseBody():
    global bodyRaisedFlag
    solenoid.off()
    bodyRaisedFlag = True


def isBodyRaised():
    return bodyRaisedFlag


def lowerBody():
    global bodyRaisedFlag
    solenoid.on()
    bodyRaisedFlag = False


def pneumatics_test():
    """Testing loop only - used for testing the pneumatics."""
    while True:
        time.sleep(10.0)
        print("solenoid on...")
        solenoid.on()
        time.sleep(10)
        print("solenoid off...")
        solenoid.off()


######################################################
# TTS CLIENT
######################################################
http = requests.Session()   # persistent connection - avoids TCP setup on every sentence


def local_effects(data, rate):
    """Fallback: original sox pitch/speed/reverb chain, run on the Pi. Only used if the
    server did not apply effects (pedalboard not installed there)."""
    import torch
    import torchaudio
    wav = torch.tensor(data).unsqueeze(0)
    effects = [
        ['pitch', str(int(PITCH_SHIFT * 100))],
        ['speed', '0.95'],          # must be before 'rate' effect in list
        ['rate', str(rate)],
        ['reverb', '20'],
    ]
    out, _ = torchaudio.sox_effects.apply_effects_tensor(wav, rate, effects)
    return out.squeeze().numpy()


def render_clip(text):
    """Text -> (float32 mono numpy array, sample_rate) with effects applied, or None on error."""
    try:
        r = http.post(
            f"{TTS_SERVER}/speak",
            json={"text": text, "speaker": TTS_SPEAKER, "effects": True, "pitch_shift": PITCH_SHIFT},
            timeout=TTS_TIMEOUT,
        )
        r.raise_for_status()
        data, rate = sf.read(io.BytesIO(r.content), dtype="float32")
        if r.headers.get("X-Effects-Applied") != "1":
            data = local_effects(data, rate)
        return data, rate
    except Exception as e:
        log("render_clip", f"TTS failed for '{text[:40]}': {e}")
        return None


def prerender(lines):
    clips = []
    for line in lines:
        for _ in range(5):                  # server may still be starting
            clip = render_clip(line)
            if clip is not None:
                clips.append(clip)
                break
            time.sleep(2)
    return clips


######################################################
# AUDIO PIPELINE:  text_q -> tts_worker -> audio_q -> player_worker
######################################################
text_q = queue.Queue()
audio_q = queue.Queue()
END = object()                      # end-of-turn marker travelling through both queues
turn_lock = threading.Lock()
turn_done = threading.Event()
turn = {"id": 0, "settled": False, "t0": None}
# 'settled' = real speech audio has been queued (or the turn ended), so a late filler must not be
# queued.  It is always checked/set under turn_lock, which guarantees the filler can never land
# behind real audio.


def begin_turn():
    with turn_lock:
        turn["id"] += 1
        turn["settled"] = False
        turn["t0"] = time.time()
        turn_done.clear()
        return turn["id"]


def tts_worker():
    while True:
        item = text_q.get()
        if item is END:
            with turn_lock:
                turn["settled"] = True
                audio_q.put(END)
            continue
        clip = render_clip(item)
        if clip is None:
            continue
        with turn_lock:
            turn["settled"] = True
            audio_q.put(("speech", clip))


def player_worker():
    while True:
        item = audio_q.get()
        if item is END:
            time.sleep(ECHO_TAIL)           # let room echo / reverb die out before the mic reopens
            turn_done.set()
            continue
        kind, (data, rate) = item
        try:
            if kind == "speech":
                t0 = turn.pop("t0", None)
                if t0:
                    log("player", f"time from end of speech to first audio: {time.time() - t0:.2f}s")
            sd.play(data, samplerate=rate)
            sd.wait()
        except Exception as e:
            print(f"Playback error: {e}")


def wait_until_quiet():
    """Block until nothing is queued for synthesis/playback and the current turn has finished."""
    deadline = time.time() + 120
    while time.time() < deadline:
        if text_q.empty() and audio_q.empty() and (turn_done.is_set() or turn["id"] == 0):
            return
        time.sleep(0.05)


def flush_mic(source):
    """Throw away audio the sound card buffered while the mic stream was open but unread.
    Without this, the next listen() starts with the creature's own voice from the last turn."""
    try:
        stream = source.stream.pyaudio_stream
        for _ in range(200):                # hard cap so this can never spin forever
            avail = stream.get_read_available()
            if avail <= 0:
                break
            stream.read(min(avail, 4096), exception_on_overflow=False)
    except Exception as e:
        log("flush_mic", f"could not flush mic buffer: {e}")


def filler_timer(turn_id, clip):
    time.sleep(FILLER_DELAY)
    with turn_lock:
        if turn["id"] == turn_id and not turn["settled"]:
            audio_q.put(("filler", clip))


def play_canned(clip):
    """Play one pre-rendered clip and wait for it to finish."""
    if clip is None:
        return
    begin_turn()
    with turn_lock:
        turn["settled"] = True
        audio_q.put(("speech", clip))
        audio_q.put(END)
    turn_done.wait(timeout=60)


def play_random(clips):
    if clips:
        play_canned(random.choice(clips))


######################################################
# GPT (streamed)
######################################################
def reset_gpt_session():
    global messages, session_id
    with history_lock:
        messages = list(messages_base)
        session_id += 1


reset_gpt_session()

SENTENCE_END = re.compile(r'(?<=[.!?\u2026])\s+')


def ask_gpt_stream(prompt):
    """Generator yielding speakable chunks (usually sentences) as GPT produces them."""
    with history_lock:
        msgs = messages
        sid = session_id
        msgs.append({"role": "user", "content": prompt})
        if len(msgs) > 1 + MAX_HISTORY:                 # sliding window; system prompt is kept
            del msgs[1:len(msgs) - MAX_HISTORY]
        snapshot = list(msgs)

    stream = client.chat.completions.create(
        model=OPENAI_MODEL, messages=snapshot, stream=True, max_tokens=MAX_TOKENS)

    buf, full, pending = "", "", ""
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            tok = chunk.choices[0].delta.content or ""
            if not tok:
                continue
            buf += tok
            full += tok
            parts = SENTENCE_END.split(buf)
            if len(parts) > 1:
                complete, buf = parts[:-1], parts[-1]
                for s in complete:
                    pending = (pending + " " + s).strip()
                    if len(pending) >= MIN_CHARS:
                        yield pending
                        pending = ""
        leftover = (pending + " " + buf).strip()
        if leftover:
            yield leftover
    finally:
        # keep conversation context - unless the visitor left and the session was reset meanwhile
        if full.strip():
            with history_lock:
                if sid == session_id:
                    msgs.append({"role": "assistant", "content": full})
        log("ask_gpt", "chatgpt response: " + full)


def respond_streaming(user_text):
    my_id = begin_turn()
    if FILLER_CLIP is not None:
        threading.Thread(target=filler_timer, args=(my_id, FILLER_CLIP), daemon=True).start()
    got_any = False
    try:
        for sentence in ask_gpt_stream(user_text):
            got_any = True
            if debugVoiceThread:
                log("speakingTask", f"queueing: {sentence}")
            text_q.put(sentence)
    except Exception as e:
        print(f"GPT error: {e}")
    if not got_any:
        text_q.put("The void is silent tonight. Try again, wanderer.")
    text_q.put(END)
    turn_done.wait(timeout=90)


######################################################
# SPEECH RECOGNITION
######################################################
def mic_meter(recognizer, source):
    """Tuning aid (MIC_TEST = True): prints live mic level vs. the speech threshold. Speak
    normally from where visitors stand - your voice should read well above the threshold and
    background noise well below it. Ctrl-C to stop."""
    print("MIC TEST - speak, then stay quiet; Ctrl-C to quit")
    while True:
        data = source.stream.read(source.CHUNK)
        level = audioop.rms(data, source.SAMPLE_WIDTH)
        bar = "#" * min(60, level // 20)
        flag = "SPEECH" if level > recognizer.energy_threshold else "      "
        print(f"{level:5d} {flag} {bar}")


def listen(recognizer, source):
    """Returns (status, text): status is 'ok', 'timeout', 'unintelligible' or 'error'.

    'timeout' is only returned after LISTEN_TIMEOUT seconds with no real speech.  Short noise
    blips (a click, a chair, speaker tail) are ignored and listening continues, so they can no
    longer trigger the 'repeat that' line instantly."""
    deadline = time.time() + LISTEN_TIMEOUT
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return "timeout", None
        try:
            audio = recognizer.listen(source, timeout=remaining, phrase_time_limit=PHRASE_LIMIT)
        except sr.WaitTimeoutError:
            return "timeout", None

        secs = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
        level = audioop.rms(audio.frame_data, audio.sample_width)
        log("listen", f"captured {secs:.2f}s, level {level}, threshold {recognizer.energy_threshold:.0f}")
        if secs < MIN_PHRASE_SECONDS:
            log("listen", "too short - treating as noise and continuing to listen")
            continue
        try:
            return "ok", recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            if secs < REPEAT_MIN_SECONDS:
                # an "um", a cough, servo noise... not worth interrupting the visitor - keep waiting
                log("listen", f"{secs:.2f}s of audio not understood - too short to be a real attempt, still listening")
                continue
            log("listen", f"heard {secs:.2f}s of audio but could not understand it")
            return "unintelligible", None
        except sr.RequestError as e:
            log("listen", f"speech service error: {e}")
            return "error", None


def speakingTask():
    global facePresent, FILLER_CLIP, listening

    if debugVoiceThread:
        log("speakingTask", "Entering function...")

    # prepping fixed responses (rendered once, played instantly)
    filler = prerender([FILLER_TEXT])
    FILLER_CLIP = filler[0] if filler else None
    silence_clips = prerender(SILENCE_LINES)
    repeat_clips = prerender(REPEAT_LINES)
    greeting_clips = prerender(GREETING_LINES) if GREET_ON_ARRIVAL else []
    log("speakingTask", "canned audio ready")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = PAUSE_THRESHOLD
    recognizer.non_speaking_duration = 0.4          # must be <= pause_threshold

    # open the microphone ONCE and calibrate ONCE (the original did both on every listen)
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        # raise the bar for what counts as "someone is speaking"; a fixed threshold also stops it
        # drifting down in a quiet moment and then triggering on every little sound
        recognizer.energy_threshold = min(max(recognizer.energy_threshold * ENERGY_MARGIN, MIN_ENERGY), MAX_ENERGY)
        recognizer.dynamic_energy_threshold = DYNAMIC_THRESHOLD
        log("speakingTask", f"mic energy threshold set to {recognizer.energy_threshold:.0f}")
        if MIC_TEST:
            mic_meter(recognizer, source)
        was_present = False

        while True:
            try:
                if not facePresent:
                    was_present = False
                    time.sleep(0.1)
                    continue

                if not was_present:
                    was_present = True
                    if greeting_clips:
                        play_random(greeting_clips)

                # never listen while the creature is talking, and never hear its leftovers
                wait_until_quiet()
                flush_mic(source)

                listening = True                # camera thread holds the servos still while we listen
                try:
                    status, text = listen(recognizer, source)
                finally:
                    listening = False
                if not facePresent:             # visitor left while we were waiting
                    continue

                if status == "ok":
                    if debugVoiceThread:
                        log("speakingTask", f"Visitor said: {text}")
                    respond_streaming(text)
                elif status == "timeout":
                    play_random(silence_clips)
                elif status == "unintelligible":
                    play_random(repeat_clips)
                else:
                    time.sleep(1.0)

                time.sleep(POST_SPEECH_PAUSE)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(0.5)


######################################################
# CAMERA / SERVO FACE FOLLOWING
######################################################
def cameraEyeMovementTask():
    global facePresent

    cv2.setNumThreads(2)        # leave CPU for the audio threads (they compete on a Pi)
    timeLastFacePresent = time.monotonic()

    if debugCameraThread:
        log("cameraEyeMovementTask", "Entering function...")

    # Set up the kit for 16-channel PCA9685
    kit = ServoKit(channels=16)

    pan_channel = 0
    tilt_channel = 1
    log("cameraEyeMovementTask", "Start by panning/tilting to defaults...")
    pan_angle = defaultPanAngle
    tilt_angle = defaultTiltAngle
    kit.servo[pan_channel].angle = pan_angle
    kit.servo[tilt_channel].angle = tilt_angle
    time.sleep(.01)

    # Load Haar Cascade for face detection
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Open the USB camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always work on the freshest frame

    if debugCameraThread:
        log("cameraEyeMovementTask", "Beginning infinite loop...")

    while True:
        try:
            if not cap.isOpened():
                log("cameraEyeMovementTask", "ERROR - camera not captured - re-attempting capture...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(0)
                continue

            ret, frame = cap.read()
            if not ret:
                log("cameraEyeMovementTask", "ERROR - frame not captured - will retry on loop...")
                time.sleep(0.1)
                continue

            frame_height, frame_width = frame.shape[:2]

            # Detect on a downscaled grayscale copy (much faster than full resolution)
            scale = frame_width / DETECT_WIDTH if frame_width > DETECT_WIDTH else 1.0
            small = cv2.resize(frame, (DETECT_WIDTH, int(frame_height / scale))) if scale > 1.0 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

            # Find the largest face and map it back to full-frame coordinates
            face_box = None
            if len(faces) > 0:
                best = max(faces, key=lambda f: f[2] * f[3])
                face_box = tuple(int(v * scale) for v in best)

            if face_box is not None:
                x, y, w, h = face_box
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)

            # Show the frame - shows even with no face present (helps debug and focus camera)
            if SHOW_PREVIEW:
                cv2.imshow('Face Tracking', frame)
                cv2.waitKey(1)  # NOTE: required to finalize showing the frame

            if face_box is not None:
                # indicate face is present so other threads know
                facePresent = True
                timeLastFacePresent = time.monotonic()

                frame_center_x = frame_width // 2
                frame_center_y = frame_height // 2

                # offset of center of face from center of frame
                offset_x = face_center_x - frame_center_x
                offset_y = face_center_y - frame_center_y

                # Normalize offset (fraction of frame size)
                norm_x = offset_x / frame_width
                norm_y = offset_y / frame_height

                # Face is close enough to center: leave the servos alone (no motion = no noise)
                deadband = LISTEN_DEADBAND if listening else IDLE_DEADBAND
                if abs(norm_x) < deadband and abs(norm_y) < deadband:
                    time.sleep(0.01)
                    continue

                # Compute angle adjustments - 'sensitivity' acts as a dampener to avoid jitter
                delta_pan = -norm_x * 120 * sensitivity
                delta_tilt = -norm_y * 120 * sensitivity

                # Do not allow jumps greater than max_angle_change
                delta_pan = max(-max_angle_change, min(max_angle_change, delta_pan))
                delta_tilt = max(-max_angle_change, min(max_angle_change, delta_tilt))

                # Update angles
                pan_angle += delta_pan
                tilt_angle += delta_tilt

                # Do not allow angles to exceed desired servo range
                pan_angle = max(min_pan_angle, min(max_pan_angle, pan_angle))
                if not isBodyRaised():
                    tilt_angle = max(90, min(max_tilt_angle, tilt_angle))
                else:
                    tilt_angle = max(min_tilt_angle, min(max_tilt_angle, tilt_angle))

                # Move servos
                kit.servo[pan_channel].angle = pan_angle
                kit.servo[tilt_channel].angle = tilt_angle

                # Raise / lower the body based on how far the head is tilted
                if isBodyRaised() and tilt_angle < 92:
                    tilt_angle += 15
                    kit.servo[tilt_channel].angle = tilt_angle
                    lowerBody()
                if not isBodyRaised() and tilt_angle > 125:
                    raiseBody()

                time.sleep(0.001)   # time between adjustments - .01 makes movement smooth, high values jumpy

            else:
                if facePresent:
                    ms_lost = (time.monotonic() - timeLastFacePresent) * 1000
                    if debugCameraThread:
                        log("cameraEyeMovementTask", "Lost track of face for " + str(int(ms_lost)) + "ms")

                    if ms_lost > FACE_LOST_MS:
                        # visitor left: end the session and go back to the rest pose (once)
                        facePresent = False
                        reset_gpt_session()
                        pan_angle = defaultPanAngle
                        tilt_angle = defaultTiltAngle
                        raiseBody()
                        kit.servo[pan_channel].angle = pan_angle
                        kit.servo[tilt_channel].angle = tilt_angle
                        time.sleep(2.00)
                else:
                    time.sleep(0.03)    # idle: no visitor, don't spin the CPU

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(0.1)


######################################################
# MAIN
######################################################
if __name__ == "__main__":
    if PNEUMATICS_TEST:
        pneumatics_test()

    threading.Thread(target=tts_worker, daemon=True, name="tts").start()
    threading.Thread(target=player_worker, daemon=True, name="player").start()
    threading.Thread(target=speakingTask, daemon=True, name="speaking").start()
    threading.Thread(target=cameraEyeMovementTask, daemon=True, name="camera").start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("main", "Shutting down...")
        raiseBody()
        cv2.destroyAllWindows()
