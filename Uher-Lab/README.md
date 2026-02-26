# Measurement of Psychomotor Speed from Multimodal Data

A commonly observed symptom of depression is an overall change in psychomotor function. For example, individuals who are experiencing a depressive episode may have a pronounced slowing in their speech, thought, and movement, or alternatively, have racing thoughts and restlessness. In this project, we investigated methods for deriving an objective measure of psychomotor speed and rhythm from speech recordings and actigraphy signals. We then evaluated the ability of this psychomotor indicator to predict response to depression treatments in a clinical sample collected by CAN-BIND (https://canbind.ca/).

## Overview

This methods tutorial has three components:

1. Deriving psychomotor features from speech data, using voice activity detection and Google's WebRTC project.
    - See: `speech_processing.ipynb`

2. Deriving psychomotor features from actigraphy data, using cepstral analysis to automatically detect walking bouts and estimate the wearer's cadence.
    - See: `actigraphy_processing.ipynb`

3. Combining the derived features into a single binary indicator of psychomotor extremeness which was able to predict depression treatment outcomes.
    - See: `multimodal_extremeness.ipynb`

Each notebook is annotated with commentary and refers only to demonstrative test data provided within this repository.

## Getting Started

### Dependencies

* This code was tested using Python 3.10.17

### Running the code

* It is recommended to create a virtual environment (venv) and install the dependencies from the `requirements.txt` provided
#### 1. Create a venv (assuming python3 points to the desired python version):
```
python3 -m venv venv
```

#### 2. Activate the venv
(Linux)
```
source venv/bin/activate
```

(Windows)
```
.\venv\Scripts\activate.bat
```

#### 3. Install dependencies:
```
pip install -r requirements.txt
```