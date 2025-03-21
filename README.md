# Sound Separation Project

## Overview

This repository contains a joint program that integrates two models: Conv-TasNet for speech separation and a Gender Detection model. The program also includes a client interface built using Streamlit, allowing users to easily interact with the models and visualize the results.

## Features

- **Conv-TasNet:** A state-of-the-art model for speech separation, capable of isolating individual speakers from a mixed audio signal.
- **Gender Detection:** A model for identifying the gender of speakers in an audio signal.
- **Streamlit Client:** A user-friendly web interface to interact with the models, upload audio files, and view the results.

## Installation

To run the program, you need to have Python and the required libraries installed. Follow these steps to set up the environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/mochamadness/Sound-Separation-Project.git
    cd Sound-Separation-Project
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To start the Streamlit client, run the following command:
```sh
streamlit run app.py
```

## Usage

1. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
2. Upload an audio file or record using microphone using the interface.
3. The application will process the audio file with Conv-TasNet to separate the speakers and then apply the Gender Detection model to identify the gender of each speaker.
4. The results, including the separated audio and detected genders, will be displayed on the web interface.

