#  NeuroFlow: Your Friendly Neighborhood Behavior Monitor

## The Problem

Traditional mental health monitoring often requires active reporting (like filling out surveys) or sharing sensitive data with third-party services. This can be intrusive and creates privacy risks. We wanted to see if we could use the ambient data already being generated in a smart home to detect meaningful behavioral shifts.

## Our Solution: NeuroFlow

NeuroFlow is a proof-of-concept dashboard that uses privacy-preserving AI to analyze smart home sensor data for anomalies that might indicate a change in mental well-being.

### How it Works

1.  **Ambient Intelligence**: The system uses passive, non-identifying data from common smart home sensors (motion, doors, bed presence, etc.). It doesn't know *who* you are, just *what* is happening.

2.  **Privacy-First AI**:
    *   **LSTM Autoencoder**: This is the core of the anomaly detection. The AI learns your personal "rhythm of life" from the sensor data. Think of it like a loyal pet that gets to know your daily routine. When your routine changes dramatically (e.g., staying in bed all day, or restlessly pacing all night), the model flags it as an "anomaly" because it deviates from your established baseline.
    *   **Federated Learning**: This is the real magic for privacy. NeuroFlow includes a simulation to show how multiple users could help improve the AI model *without ever sharing their personal data*. It's like a group of chefs collaborating on a better cake recipe by only sharing their baking techniques, not the secret ingredients of their own cake. Your data stays on your device, always.

3.  **Interactive Dashboard**: We use **Streamlit** to create a simple, interactive web app where you can see the data, train the model, and explore the results.

### Key Features

*   **Visualize Data**: See a full week of simulated sensor data to understand what "normal" and "anomalous" behavior looks like.
*   **Train Your Own AI**: Interactively train the LSTM autoencoder on the "normal" data.
*   **Detect Anomalies**: Run the trained model on new data to see it flag unusual patterns in real-time.
*   **Federated Learning Demo**: Run a simulation to understand how collaborative, privacy-preserving model training works.
*   **Privacy Controls**: A dedicated page to show what privacy-focused features could look like.

## Getting Started

Ready to give it a spin?

1.  **Install Dependencies**: Make sure you have Python, then open your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App**:
    ```bash
    streamlit run neuroflow_app.py
    ```
    A new tab should open in your web browser. If not, the terminal will give you a URL to click on.

    **Note**: The app will automatically generate its own synthetic data (`normal_week.csv`, `anomalous_week.csv`) on the first run, so you don't need to hunt for any data!

## What's in the Box? (File Structure)

Here's a quick look at the most important files:

*   `neuroflow_app.py`: The main brain of the Streamlit application. It handles the UI and navigation.
*   `model.py`: Defines the Keras/TensorFlow architecture for the LSTM Autoencoder.
*   `train_and_detect.py`: Contains all the logic for training the model and using it to detect anomalies.
*   `federated_learning.py`: Holds the code for the federated learning simulation.
*   `data_generator.py`: The script that creates the fake sensor data for the demo.
*   `requirements.txt`: The list of Python libraries needed to run the project.

---

*Disclaimer: This is a hackathon project. If it gains sentience and starts judging your life choices, that was not part of the original plan. Please unplug it immediately.*
