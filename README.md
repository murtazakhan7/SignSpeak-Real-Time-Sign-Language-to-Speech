
# SignSpeak: Real-Time Sign Language to Speech

SignSpeak is an interactive deep learning project that bridges the communication gap between sign language users and non-signers. It uses a fine-tuned CNN with **ResNet**, **Mediapipe** for hand detection, and **Streamlit** for a real-time interface. The system detects sign language letters from live webcam input or uploaded images, displays them on screen, and converts them into audible speech.

## Features

* **Real-Time Detection**: Detects hand signs from webcam using Mediapipe hand tracking.
* **Image Upload Support**: Classify signs from uploaded images.
* **Speech Output**: Converts predicted letters into audible speech via `pyttsx3`.
* **Interactive UI**: Built with Streamlit for easy use and visualization.
* **Transfer Learning**: Fine-tuned ResNet on custom dataset for sign language classification (36 classes).

---

## Tech Stack

* **Deep Learning**: TensorFlow, Keras (ResNet, CNN layers)
* **Computer Vision**: OpenCV, Mediapipe
* **Frontend**: Streamlit
* **Speech**: Pyttsx3 (offline TTS)
* **Language**: Python

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SignSpeak.git
cd SignSpeak
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app/streamlit_app.py
```

## Usage

* **Webcam Mode**: Start webcam, show hand signs → predictions appear on screen & spoken out loud.
* **Upload Mode**: Upload an image of a hand sign → system predicts and speaks the result.



## Future Work

* Extend model to recognize **word-level signs**.
* Add support for **“space”** and **“end”** gestures to form complete sentences.
* Integrate advanced TTS for more natural speech.

