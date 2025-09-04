"""
NeuroFlow Model Architecture
============================
This module contains the LSTM Autoencoder architecture for anomaly detection
and data preprocessing functions for time series analysis.

Author: NeuroFlow Team
Date: 2025
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def create_lstm_autoencoder(timesteps, features, latent_dim=50):
    """
    Create LSTM Autoencoder architecture for anomaly detection.

    The autoencoder learns to reconstruct normal behavioral patterns and
    flags anomalies when reconstruction error exceeds learned thresholds.

    Parameters:
    -----------
    timesteps : int
        Number of time steps in input sequences (e.g., 24 for 24-hour windows)
    features : int  
        Number of sensor features (e.g., 5 for motion, door, bed, steps, sleep)
    latent_dim : int, default=50
        Dimensionality of the encoded representation (bottleneck layer)

    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM Autoencoder model ready for training
    """

    # Input layer for time series data
    encoder_inputs = layers.Input(shape=(timesteps, features), name='encoder_input')

    # Encoder: Compress temporal patterns into latent representation
    # First LSTM layer with dropout for regularization
    encoder_lstm1 = layers.LSTM(
        latent_dim * 2, 
        activation='relu', 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name='encoder_lstm1'
    )(encoder_inputs)

    # Second LSTM layer (bottleneck)
    encoder_lstm2 = layers.LSTM(
        latent_dim, 
        activation='relu',
        dropout=0.2,
        recurrent_dropout=0.2,
        name='encoder_lstm2'
    )(encoder_lstm1)

    # Decoder: Reconstruct original time series from latent representation
    # Repeat the encoded vector for each timestep
    decoder_repeat = layers.RepeatVector(timesteps, name='decoder_repeat')(encoder_lstm2)

    # First decoder LSTM layer
    decoder_lstm1 = layers.LSTM(
        latent_dim, 
        activation='relu', 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name='decoder_lstm1'
    )(decoder_repeat)

    # Second decoder LSTM layer
    decoder_lstm2 = layers.LSTM(
        latent_dim * 2, 
        activation='relu', 
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        name='decoder_lstm2'
    )(decoder_lstm1)

    # Output layer: Reconstruct original features for each timestep
    decoder_outputs = layers.TimeDistributed(
        layers.Dense(features, activation='linear'), 
        name='decoder_output'
    )(decoder_lstm2)

    # Create the complete autoencoder model
    autoencoder = Model(encoder_inputs, decoder_outputs, name='lstm_autoencoder')

    # Compile with Adam optimizer and mean squared error loss
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return autoencoder

def create_encoder_model(autoencoder):
    """
    Extract the encoder portion from a trained autoencoder.

    Parameters:
    -----------
    autoencoder : tensorflow.keras.Model
        Trained LSTM autoencoder model

    Returns:
    --------
    tensorflow.keras.Model
        Encoder model that outputs latent representations
    """

    encoder_input = autoencoder.input
    encoder_output = autoencoder.get_layer('encoder_lstm2').output
    encoder_model = Model(encoder_input, encoder_output, name='lstm_encoder')

    return encoder_model

def prepare_sequences(data, timesteps=24, feature_columns=None):
    """
    Prepare time series data for LSTM autoencoder training.

    This function:
    1. Normalizes the data using MinMaxScaler
    2. Creates sliding window sequences of specified length
    3. Returns both sequences and the fitted scaler for inverse transformation

    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with timestamp and sensor columns
    timesteps : int, default=24
        Length of each sequence (e.g., 24 hours)
    feature_columns : list, optional
        List of column names to use as features. If None, excludes 'timestamp'

    Returns:
    --------
    tuple
        (sequences, scaler) where sequences is numpy array of shape 
        (num_sequences, timesteps, features) and scaler is fitted MinMaxScaler
    """

    # Select feature columns (exclude timestamp by default)
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != 'timestamp']

    # Extract feature data
    feature_data = data[feature_columns].values

    # Normalize features to [0, 1] range for stable LSTM training
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)

    # Create sliding window sequences
    sequences = []
    for i in range(len(scaled_data) - timesteps + 1):
        sequence = scaled_data[i:i + timesteps]
        sequences.append(sequence)

    sequences = np.array(sequences)

    print(f"📊 Data preparation complete:")
    print(f"   Original data shape: {feature_data.shape}")
    print(f"   Scaled data shape: {scaled_data.shape}")
    print(f"   Sequences shape: {sequences.shape}")
    print(f"   Features used: {feature_columns}")

    return sequences, scaler

def calculate_reconstruction_error(model, sequences):
    """
    Calculate reconstruction error for anomaly detection.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained LSTM autoencoder
    sequences : numpy.ndarray
        Input sequences to reconstruct

    Returns:
    --------
    numpy.ndarray
        Reconstruction errors for each sequence
    """

    # Get model predictions (reconstructions)
    predictions = model.predict(sequences, verbose=0)

    # Calculate mean squared error for each sequence
    mse_errors = np.mean(np.square(sequences - predictions), axis=(1, 2))

    return mse_errors

def determine_threshold(errors, percentile=95):
    """
    Determine anomaly detection threshold based on reconstruction errors.

    Parameters:
    -----------
    errors : numpy.ndarray
        Reconstruction errors from normal data
    percentile : float, default=95
        Percentile to use for threshold (e.g., 95 means top 5% are anomalies)

    Returns:
    --------
    float
        Threshold value for anomaly detection
    """

    threshold = np.percentile(errors, percentile)

    print(f"📊 Threshold Analysis:")
    print(f"   Mean reconstruction error: {np.mean(errors):.6f}")
    print(f"   Std reconstruction error: {np.std(errors):.6f}")  
    print(f"   {percentile}th percentile threshold: {threshold:.6f}")

    return threshold

def detect_anomalies(model, sequences, threshold):
    """
    Detect anomalies in time series data using trained autoencoder.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained LSTM autoencoder
    sequences : numpy.ndarray
        Input sequences to analyze
    threshold : float
        Anomaly detection threshold

    Returns:
    --------
    tuple
        (anomalies, errors) where anomalies is boolean array and errors is float array
    """

    # Calculate reconstruction errors
    errors = calculate_reconstruction_error(model, sequences)

    # Identify anomalies (errors above threshold)
    anomalies = errors > threshold

    print(f"🔍 Anomaly Detection Results:")
    print(f"   Total sequences analyzed: {len(sequences)}")
    print(f"   Anomalies detected: {np.sum(anomalies)}")
    print(f"   Anomaly rate: {np.sum(anomalies)/len(anomalies)*100:.2f}%")

    return anomalies, errors

def save_model_and_threshold(model, threshold, model_path='neuroflow_autoencoder.h5', 
                           threshold_path='threshold.npy'):
    """
    Save trained model and threshold for later use.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained autoencoder model
    threshold : float
        Anomaly detection threshold
    model_path : str
        Path to save the model
    threshold_path : str
        Path to save the threshold
    """

    # Save model
    model.save(model_path)

    # Save threshold
    np.save(threshold_path, threshold)

    print(f"💾 Model and threshold saved:")
    print(f"   Model: {model_path}")
    print(f"   Threshold: {threshold_path}")

def load_model_and_threshold(model_path='neuroflow_autoencoder.h5', 
                           threshold_path='threshold.npy'):
    """
    Load trained model and threshold from disk.

    Parameters:
    -----------
    model_path : str
        Path to saved model
    threshold_path : str  
        Path to saved threshold

    Returns:
    --------
    tuple
        (model, threshold)
    """

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load threshold
    threshold = np.load(threshold_path)

    print(f"📂 Model and threshold loaded:")
    print(f"   Model: {model_path}")
    print(f"   Threshold: {threshold}")

    return model, threshold

# Model architecture summary helper
def print_model_summary(model):
    """
    Print detailed model architecture summary.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        Model to summarize
    """

    print("🧠 NeuroFlow LSTM Autoencoder Architecture:")
    print("=" * 50)
    model.summary()

    print(f"Model Parameters:")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

if __name__ == "__main__":
    # Example usage and testing
    print("🧠 NeuroFlow Model Architecture Test")
    print("=" * 40)

    # Create a sample model
    model = create_lstm_autoencoder(timesteps=24, features=5, latent_dim=32)
    print_model_summary(model)

    # Test with dummy data
    dummy_data = np.random.random((100, 24, 5))
    print(f"Testing with dummy data shape: {dummy_data.shape}")

    # Test prediction
    predictions = model.predict(dummy_data[:1], verbose=0)
    print(f"Prediction output shape: {predictions.shape}")

    print("✅ Model architecture test complete!")
