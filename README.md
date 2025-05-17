# AttendEase: Advanced Attendance System with Modern UI

AttendEase is an advanced attendance system using OpenCV for real-time face detection and recognition, now featuring a modern, creative UI with:
- Neon animated overlays
- Semi-transparent HUD and watermark
- Cross-platform speech feedback
- Live progress bars and stats

## Features
- Real-time face detection and recognition
- Automatic attendance marking
- Modern, visually appealing GUI (OpenCV-based)
- Animated overlays and creative effects
- Cross-platform speech feedback (using pyttsx3)
- Robust camera selection and troubleshooting
- Database management for face encodings and attendance records

## Installation
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd AttendEase-main
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   # Or manually:
   pip install opencv-python numpy scikit-learn pyttsx3
   ```
3. Ensure you have a webcam and grant camera permissions to your terminal or IDE (see below).

## Usage
1. **Collect Face Data:**
   ```sh
   python3 main.py
   ```
   - Enter your name when prompted.
   - Look at the camera until 100 samples are collected (progress bar will show).

2. **Run Attendance System:**
   ```sh
   python3 testing.py
   ```
   - The system will recognize faces and mark attendance.
   - Press 'm' to mark attendance and hear speech feedback.

## Camera Troubleshooting
- If you see "Camera not accessible!", ensure:
  - Your webcam is connected and not used by another app.
  - You have granted camera access to your terminal/IDE in System Settings > Privacy & Security > Camera.
  - The app will auto-select the first available camera index (0-3).

## Attendance Records
- Attendance is saved as CSV files in the `Attendance` directory, one file per day.

## Credits
- Built with OpenCV, scikit-learn, numpy, and pyttsx3.
- UI inspired by modern HUD and neon design trends.

---
Enjoy your futuristic attendance system!
