import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import mediapipe as mp
import pyttsx3
import time
from collections import deque
import threading

# --------------------
# Configuration
# --------------------
IMG_SIZE = (300, 300)  # Match training size (updated from 224 to 300)
MODEL_PATH = "sign_language_final_optimized.keras"  # Updated model path
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction

# ASL Alphabet + Numbers (0-9) = 36 classes
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# Special commands for word formation
SPACE_GESTURE = "SPACE"  # You can assign a specific gesture
DELETE_GESTURE = "DELETE"

# --------------------
# Page Configuration
# --------------------
st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .word-display {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        font-size: 1.8rem;
        background: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 30px;
        margin: 10px 0;
    }
    .stats-box {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------
# Initialize Session State
# --------------------
if 'current_word' not in st.session_state:
    st.session_state.current_word = ""
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'prediction_stable_count' not in st.session_state:
    st.session_state.prediction_stable_count = 0
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

# --------------------
# Text-to-Speech Setup
# --------------------
@st.cache_resource
def init_tts():
    """Initialize TTS engine with error handling"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # Set voice (try to get a clear voice)
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)  # Often female voice
        
        return engine
    except Exception as e:
        st.warning(f"TTS initialization failed: {e}")
        return None

tts_engine = init_tts()
tts_lock = threading.Lock()

def speak_async(text):
    """Speak text in separate thread to avoid blocking"""
    if tts_engine is None or not text:
        return
    
    def _speak():
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                pass  # Silent fail for TTS
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()

# --------------------
# Load Model
# --------------------
@st.cache_resource
def load_sign_model():
    """Load the trained model with error handling"""
    try:
        model = load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_sign_model()

if error:
    st.error(f"❌ Failed to load model: {error}")
    st.info("Please ensure the model file exists at the specified path.")
    st.stop()

# --------------------
# MediaPipe Hand Detection
# --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# --------------------
# Image Processing Functions
# --------------------
def extract_hand_region(image, padding=40):
    """
    Detect hand using MediaPipe and return cropped hand image with landmarks.
    Returns: (cropped_image, landmarks, bbox) or (None, None, None)
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        xmin = max(0, int(min(x_coords) * w) - padding)
        xmax = min(w, int(max(x_coords) * w) + padding)
        ymin = max(0, int(min(y_coords) * h) - padding)
        ymax = min(h, int(max(y_coords) * h) + padding)
        
        # Crop hand region
        cropped = image[ymin:ymax, xmin:xmax]
        
        # Make square for better aspect ratio
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            max_dim = max(cropped.shape[0], cropped.shape[1])
            square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            y_offset = (max_dim - cropped.shape[0]) // 2
            x_offset = (max_dim - cropped.shape[1]) // 2
            square[y_offset:y_offset+cropped.shape[0], 
                   x_offset:x_offset+cropped.shape[1]] = cropped
            
            return square, hand_landmarks, (xmin, ymin, xmax, ymax)
    
    return None, None, None

def preprocess_for_model(image):
    """Preprocess image to match training pipeline"""
    # Resize to model input size
    image = cv2.resize(image, IMG_SIZE)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize using EfficientNet preprocessing
    image = image.astype("float32")
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_sign(image):
    """
    Predict sign language character with confidence.
    Returns: (predicted_class, confidence, all_predictions)
    """
    processed = preprocess_for_model(image)
    predictions = model.predict(processed, verbose=0)[0]
    
    # Get top prediction
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    predicted_class = CLASS_NAMES[class_id]
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3_predictions = [(CLASS_NAMES[i], predictions[i]) for i in top3_indices]
    
    return predicted_class, confidence, top3_predictions

def stabilize_prediction(current_pred, confidence, threshold=3):
    """
    Stabilize predictions to avoid flickering.
    Only accept prediction if it's consistent for 'threshold' frames.
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return None
    
    if current_pred == st.session_state.last_prediction:
        st.session_state.prediction_stable_count += 1
    else:
        st.session_state.last_prediction = current_pred
        st.session_state.prediction_stable_count = 1
    
    if st.session_state.prediction_stable_count >= threshold:
        return current_pred
    
    return None

def add_to_word(character):
    """Add character to current word"""
    st.session_state.current_word += character
    st.session_state.prediction_history.append({
        'char': character,
        'time': time.time()
    })
    st.session_state.total_predictions += 1
    
    # Speak the character
    speak_async(character)

def add_space():
    """Add current word to sentence and start new word"""
    if st.session_state.current_word:
        st.session_state.sentence += st.session_state.current_word + " "
        speak_async(st.session_state.current_word)
        st.session_state.current_word = ""

def delete_last_char():
    """Delete last character from current word"""
    if st.session_state.current_word:
        st.session_state.current_word = st.session_state.current_word[:-1]

# --------------------
# UI Components
# --------------------
def draw_landmarks_on_frame(frame, landmarks, bbox):
    """Draw hand landmarks on frame"""
    if landmarks and bbox:
        # Draw landmarks
        h, w, _ = frame.shape
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw bounding box
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return frame

# --------------------
# Main UI
# --------------------
st.markdown('<p class="main-header">🤟 Sign Language Translator</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    mode = st.radio("**Mode:**", ["📷 Live Webcam", "📁 Upload Image"])
    
    st.markdown("---")
    
    # Display settings
    show_landmarks = st.checkbox("Show Hand Landmarks", value=True)
    show_confidence = st.checkbox("Show Confidence Bar", value=True)
    show_top3 = st.checkbox("Show Top 3 Predictions", value=True)
    
    st.markdown("---")
    
    # Word formation controls
    st.subheader("📝 Word Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Space", use_container_width=True):
            add_space()
    with col2:
        if st.button("⬅️ Delete", use_container_width=True):
            delete_last_char()
    
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.current_word = ""
        st.session_state.sentence = ""
        st.session_state.prediction_history = []
        st.rerun()
    
    if st.button("🔊 Speak Sentence", use_container_width=True):
        full_text = st.session_state.sentence + st.session_state.current_word
        if full_text.strip():
            speak_async(full_text)
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Characters", st.session_state.total_predictions)
    st.metric("Current Word Length", len(st.session_state.current_word))
    st.metric("Words Formed", len(st.session_state.sentence.split()))

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Camera/Image View")

with col2:
    st.subheader("📝 Formed Text")
    
    # Display current sentence
    if st.session_state.sentence:
        st.markdown(f'<div class="word-display">{st.session_state.sentence}</div>', 
                   unsafe_allow_html=True)
    
    # Display current word being formed
    if st.session_state.current_word:
        st.markdown(f'<div class="prediction-box">Current: {st.session_state.current_word}</div>', 
                   unsafe_allow_html=True)
    else:
        st.info("Start signing to form words!")

# --------------------
# Upload Image Mode
# --------------------
if mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a hand sign image", 
                                     type=["jpg", "jpeg", "png"],
                                     help="Upload a clear image of a hand sign")
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process image
        hand_img, landmarks, bbox = extract_hand_region(image_bgr)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if hand_img is not None:
                # Draw landmarks if enabled
                display_img = image_bgr.copy()
                if show_landmarks:
                    display_img = draw_landmarks_on_frame(display_img, landmarks, bbox)
                
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), 
                        caption="Detected Hand", use_column_width=True)
                
                # Make prediction
                predicted_class, confidence, top3 = predict_sign(hand_img)
                
                with col2:
                    # Display main prediction
                    st.markdown(f'<div class="prediction-box">{predicted_class}</div>', 
                               unsafe_allow_html=True)
                    
                    # Confidence bar
                    if show_confidence:
                        st.progress(float(confidence))
                        st.caption(f"Confidence: {confidence*100:.1f}%")
                    
                    # Top 3 predictions
                    if show_top3:
                        st.markdown("**Top 3 Predictions:**")
                        for i, (cls, conf) in enumerate(top3, 1):
                            st.write(f"{i}. {cls}: {conf*100:.1f}%")
                    
                    # Add to word button
                    if confidence >= CONFIDENCE_THRESHOLD:
                        if st.button(f"➕ Add '{predicted_class}' to word", use_container_width=True):
                            add_to_word(predicted_class)
                            st.rerun()
                    else:
                        st.warning(f"Low confidence ({confidence*100:.1f}%). Need {CONFIDENCE_THRESHOLD*100:.0f}%+")
            else:
                st.error("❌ No hand detected in the image!")
                st.info("💡 Tips:\n- Ensure good lighting\n- Position hand clearly in frame\n- Try different angles")
                st.image(image, caption="Original Image", use_column_width=True)

# --------------------
# Live Webcam Mode
# --------------------
elif mode == "📷 Live Webcam":
    st.write("**Instructions:**")
    st.info("✋ Show a sign → Hold steady for 3 frames → Character auto-adds to word")
    
    run = st.checkbox("▶️ Start Webcam", key="webcam_toggle")
    
    FRAME_WINDOW = st.empty()
    prediction_area = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        last_added_char = ""
        last_add_time = 0
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Please check permissions.")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            frame_count += 1
            
            # Extract hand
            hand_img, landmarks, bbox = extract_hand_region(frame)
            
            if hand_img is not None:
                # Predict
                predicted_class, confidence, top3 = predict_sign(hand_img)
                
                # Stabilize prediction
                stable_pred = stabilize_prediction(predicted_class, confidence)
                
                # Draw landmarks
                if show_landmarks:
                    frame = draw_landmarks_on_frame(frame, landmarks, bbox)
                
                # Display prediction on frame
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.putText(frame, f"{predicted_class} ({confidence*100:.0f}%)", 
                           (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Stability indicator
                stability = st.session_state.prediction_stable_count
                cv2.rectangle(frame, (30, 80), (30 + stability * 40, 100), (0, 255, 0), -1)
                
                # Auto-add to word if stable and enough time passed
                if stable_pred and stable_pred != last_added_char:
                    if (time.time() - last_add_time) > 2.0:  # 2 second cooldown
                        add_to_word(stable_pred)
                        last_added_char = stable_pred
                        last_add_time = time.time()
                        
                        # Visual feedback
                        cv2.putText(frame, f"Added: {stable_pred}!", 
                                   (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "No Hand Detected", (30, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                st.session_state.prediction_stable_count = 0
            
            # Display current word on frame
            if st.session_state.current_word:
                cv2.putText(frame, f"Word: {st.session_state.current_word}", 
                           (30, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Show frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Check if user stopped
            if not st.session_state.get('webcam_toggle', False):
                break
        
        cap.release()
    else:
        st.info("👆 Check the box above to start the webcam")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 <b>Tips:</b> Good lighting, clear hand positioning, and steady gestures improve accuracy!</p>
    <p>🔊 Audio feedback enabled | 📊 Real-time confidence tracking | 🎯 36 classes (A-Z, 0-9)</p>
</div>
""", unsafe_allow_html=True)
