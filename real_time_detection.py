import cv2
import numpy as np
import tensorflow as tf
import keyboard
import time

# Load the trained CNN model
MODEL_PATH = r"D:\ARTIFICAL INTELLIGENCE\SEM 2\wavepointer\models\h5\cnn"
model = tf.keras.models.load_model(MODEL_PATH)

# Gesture labels (10 VLC gestures)
GESTURE_CLASSES = [
    "Play/Pause", "Volume Up", "Volume Down", "Next Track",
    "Previous Track", "Mute", "Stop", "Increase Brightness",
    "Decrease Brightness", "Fullscreen Toggle"
]

# VLC media control mapping (10 gestures)
VLC_CONTROL_ACTIONS = {
    0: "space",         # Play/Pause
    1: "volumeup",      # Volume Up
    2: "volumedown",    # Volume Down
    3: "ctrl+right",    # Next Track
    4: "ctrl+left",     # Previous Track
    5: "m",             # Mute
    6: "s",             # Stop
    7: "fn+f3",         # Increase Brightness
    8: "fn+f2",         # Decrease Brightness
    9: "f"              # Toggle Fullscreen
}

# Track last performed gesture
last_gesture = None
gesture_cooldown = 2  # Prevents repeated actions (in seconds)
last_action_time = time.time()

def preprocess_frame(frame):
    """Preprocess the frame for model prediction."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (128, 128))  # Resize to match model input
    frame = frame / 255.0  # Normalize pixel values
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension for grayscale
    return frame

def control_media(gesture_id):
    """Perform media control action using keyboard library, preventing repeated actions."""
    global last_gesture, last_action_time
    action = VLC_CONTROL_ACTIONS.get(gesture_id)

    # Prevent repeated actions within cooldown period
    if action and (last_gesture != gesture_id or (time.time() - last_action_time) > gesture_cooldown):
        keyboard.press_and_release(action)  # Simulate key press
        last_gesture = gesture_id  # Store last performed gesture
        last_action_time = time.time()  # Update last action time
        print(f"✅ Gesture Detected: {GESTURE_CLASSES[gesture_id]} → Action: {action}")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error accessing the camera.")
        break

    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    # Prevent index out of range error
    if predicted_label >= len(GESTURE_CLASSES):
        gesture_text = "Unknown Gesture"
    else:
        gesture_text = f"Gesture: {GESTURE_CLASSES[predicted_label]} ({confidence * 100:.2f}%)"

    # Display detected gesture
    cv2.putText(frame, gesture_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("Gesture Recognition (VLC Control)", frame)

    # Perform media control action if confidence is high
    if confidence > 0.8 and predicted_label < len(GESTURE_CLASSES):
        control_media(predicted_label)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
