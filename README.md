# AI-ML Based Intelligent De-Smoking/De-Hazing Algorithm

## Project Overview
This project focuses on developing an **AI-ML-based de-smoking/de-hazing algorithm** to enhance visibility in hazy or smoky conditions. It utilizes deep learning to process hazy images and videos, providing clear outputs for better visualization and analysis. The system integrates neural network architectures, live video processing, and real-time dehazing to achieve its goals.

This project was selected among the **top 5 teams nationwide in the Smart India Hackathon 2023** for the **Ministry of Defence's problem statement (PS ID: SIH1417)**.

---

## Methodology
### Idea, Solution, and Prototype:
1. **Neural Network Architecture:**
   - Designed two models:
     - A convolutional neural network (CNN) with feature concatenation for dehazing.
     - A network using residual connections for capturing fine details.
2. **Training Neural Networks:**
   - Trained the models using paired image datasets (hazy and clear images).
3. **Live Video Streaming Integration:**
   - Utilized OpenCV to capture and process live video streams frame by frame.
4. **Processing Hazy Test Images:**
   - Applied trained models to hazy frames for dehazing.
5. **Retrieve Clear Image Stream:**
   - **Real-time:** Stream both hazy and dehazed frames side-by-side.
   - **Video Reconstruction:** Recombine dehazed frames into a clear output video.

---

## Features
- **Dehazing Algorithms:**
  - Convolutional layers with feature concatenation.
  - Residual connections for detail preservation.
- **Real-Time Processing:**
  - Integration with live video streams using OpenCV.
- **Video Reconstruction:**
  - Frame-by-frame dehazing and video output generation.
- **Customizable Framework:**
  - Flexible architecture for various input resolutions and datasets.

---

## Tools and Technologies
- **Programming Language:** Python
- **Frameworks and Libraries:** TensorFlow, OpenCV, NumPy, Matplotlib
- **Model Training:** CNNs with feature concatenation and residual connections
- **Video Processing:** OpenCV for real-time frame handling

---

## Data and References
This project utilized datasets and insights from the repository [All-In-One-Image-Dehazing-Tensorflow](https://github.com/tusharsircar95/All-In-One-Image-Dehazing-Tensorflow).

---