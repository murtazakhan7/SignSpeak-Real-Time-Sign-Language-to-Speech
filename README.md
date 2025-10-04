
# SignSpeak: Real-Time Sign Language to Speech

SignSpeak is an interactive deep learning project that bridges the communication gap between sign language users and non-signers. It uses a fine-tuned CNN with **ResNet**, **Mediapipe** for hand detection, and **Streamlit** for a real-time interface. The system detects sign language letters from live webcam input or uploaded images, displays them on screen, and converts them into audible speech.

## Features

* **Real-Time Detection**: Detects hand signs from webcam using Mediapipe hand tracking.
* **Image Upload Support**: Classify signs from uploaded images.
* **Speech Output**: Converts predicted letters into audible speech via `pyttsx3`.
* **Interactive UI**: Built with Streamlit for easy use and visualization.
* **Transfer Learning**: Fine-tuned ResNet on custom dataset for sign language classification (36 classes).

---
Two-Phase Transfer Learning: From 60% to 98% Accuracy
The Dramatic Performance Jump
Your training achieved remarkable results through a strategic two-phase approach:
Phase 1 (Warmup - 9.2 minutes):

Started: Random weights on custom head
Ended: 80.9% validation accuracy
Base model: Completely frozen (using pre-trained ImageNet features)

Phase 2 (Fine-tuning - 13 minutes):

Started: 80.9% → Ended: 97.8% validation accuracy
Base model: Last 80 layers unfrozen
Improvement: +17 percentage points

Why This Worked So Well
1. Proper Foundation Building
Phase 1 trained a good classification head without corrupting the pre-trained features. Starting at 60% and reaching 81% showed the head learned meaningful patterns before touching the base model.
2. Gentle Fine-tuning
Using a 20x lower learning rate (5e-5 vs 1e-3) in Phase 2 prevented catastrophic forgetting. The pre-trained features adapted gradually to sign language specifics without losing their general image understanding.
3. Label Smoothing Impact
The 0.1 label smoothing prevented overconfidence. Notice your loss values (0.84-0.93) are higher than typical, but accuracy is excellent - this indicates well-calibrated, generalizable predictions.
4. Perfect Top-K Performance

Top-3 Accuracy: 100% (correct sign always in top 3 predictions)
Top-5 Accuracy: 100%

This shows the model has strong understanding of visually similar signs (like 'n' vs 'm', 'w' vs '6').
The Weak Spots
Your confusion matrix reveals specific challenging pairs:

'w' → '6': 2 errors (33% failure rate)
'n' → 'm': 2 errors (40% failure rate)
'z': 33% accuracy (confused with 'd' and '1')

These are visually similar signs where finger positions differ subtly.
Time Efficiency
22.6 minutes total for 45 epochs on ~4,000 images is exceptional. The two-phase approach was actually faster than training everything from scratch would have been.
Bottom Line
Two-phase training delivered professional-grade results (97.8%) in under 25 minutes by:

Building a strong head first
Carefully adapting pre-trained features
Using aggressive regularization (label smoothing, dropout, augmentation)

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

