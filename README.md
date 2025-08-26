# Joint Angle Estimation Application for Rehabilitation Center

## Overview
This web application is designed to assist rehabilitation centers in monitoring and improving patients' arm and shoulder movements. It detects real-time movements using MediaPipe and calculates joint angles to evaluate whether a patient achieves target positions during therapy.

## Features
- Real-time movement detection for:
  - Left and right arms
  - Left and right shoulders
- Angle calculation for joints based on specific landmarks:
  - Right arm: landmarks 12, 14, 16 (angle at 14)
  - Left arm: landmarks 11, 13, 15 (angle at 13)
  - Right shoulder: landmarks 11, 12, 14 (angle at 12)
  - Left shoulder: landmarks 12, 11, 13 (angle at 11)
- Tracks time taken to reach target angles
- Simple, user-friendly interface for easy operation in rehabilitation sessions

## Tech Stack
- Frontend: HTML, CSS
- Backend: Python (Flask)
- Framework: MediaPipe for real-time pose estimation


