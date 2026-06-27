# Camera-Based Engagement and Demographic Analysis

A graduation thesis project focused on real-time viewer engagement detection and demographic analysis using computer vision techniques.

This system analyzes whether a person is facing the camera, measures viewing duration, classifies engagement level, and generates statistical reports without storing raw camera images.

## Overview

The project aims to develop a privacy-aware camera-based analysis system that can be used in environments such as digital advertising screens, public information displays, and customer analytics systems.

The system processes webcam input in real time and evaluates viewer attention by analyzing face orientation. Based on how long a person remains engaged with the camera, the system categorizes engagement into different levels. In addition, demographic information such as age group and gender can be estimated to support statistical reporting.

## Key Features

* Real-time camera input processing
* Face detection and orientation analysis
* Viewer engagement duration tracking
* Engagement level classification
* Age group and gender prediction support
* CSV/log-based statistical reporting
* Privacy-aware design without storing raw image frames
* Modular Python project structure

## Engagement Classification

The engagement level is estimated based on the amount of time a viewer remains facing the camera.

| Duration      | Engagement Level  |
| ------------- | ----------------- |
| 0–3 seconds   | Low Engagement    |
| 3–6 seconds   | Medium Engagement |
| 6–10+ seconds | High Engagement   |

If the viewer is not facing the camera, the system classifies the state as disengaged.

## System Architecture

The general pipeline of the system is as follows:

```text
Camera Input
     ↓
Face Detection
     ↓
Face Orientation Analysis
     ↓
Engagement Time Tracking
     ↓
Demographic Prediction
     ↓
Statistical Logging and Reporting
```

## Technologies Used

* Python
* OpenCV
* MediaPipe
* scikit-learn
* Pandas
* Streamlit
* PyTorch
* Git & GitHub

## Project Structure

```text
camera-based-engagement-demographic-analysis/
│
├── src/                  # Main source code files
├── models/               # Model-related files and instructions
├── reports/              # Generated reports and sample outputs
├── data/                 # Processed or sample data files
├── training/             # Training scripts and experiments
├── dashboard/            # Dashboard or visualization interface
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files and folders
└── README.md             # Project documentation
```

## Installation

Clone the repository:

```bash
git clone https://github.com/fatmanurgencdogan/camera-based-engagement-demographic-analysis.git
cd camera-based-engagement-demographic-analysis
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main application or Streamlit dashboard depending on the project entry point:

```bash
python src/main_app_tracking_landmark_rule.py
```

or:

```bash
streamlit run src/dashboard.py
```

The exact command may vary depending on the final file structure.

## Privacy Approach

This project follows a privacy-aware design principle. Camera frames are processed in memory and are not permanently stored. The system only records statistical outputs such as timestamp, engagement status, engagement duration, predicted age group, and predicted gender.

## Thesis Information

**Project Title:** Camera-Based Engagement and Demographic Analysis
**Type:** Graduation Thesis Project
**Department:** Computer Engineering
**University:** Çukurova University

## Author

**Fatma Nur Gençdoğan**,
Computer Engineering Student,
Çukurova University

## Advisor

**Assoc. Prof. Dr. Serkan Kartal**

## License

This project is developed for academic purposes. License information can be added later depending on the publication status of the project.
