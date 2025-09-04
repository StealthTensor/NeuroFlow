"""
NeuroFlow Training and Detection Pipeline
=========================================
This module handles the complete workflow for training the LSTM Autoencoder
and performing anomaly detection on mental health sensor data.

Author: NeuroFlow Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import our custom modules
from data_generator import generate_sensor_data
from model import (
    create_lstm_autoencoder, 
    prepare_sequences, 
    calculate_reconstruction_error,
    determine_threshold,
    detect_anomalies,
    save_model_and_threshold,
    print_model_summary
)

def load_or_generate_data():
    """
    Load existing CSV data or generate new synthetic datasets.

    Returns:
    --------
    tuple
        (normal_data, anomalous_data) pandas DataFrames
    """

    try:
        # Try to load existing data
        normal_data = pd.read_csv('normal_week.csv')
        anomalous_data = pd.read_csv('anomalous_week.csv')

        # Convert timestamp column to datetime
        normal_data['timestamp'] = pd.to_datetime(normal_data['timestamp'])
        anomalous_data['timestamp'] = pd.to_datetime(anomalous_data['timestamp'])

        print("📂 Loaded existing data from CSV files")

    except FileNotFoundError:
        # Generate new data if files don't exist
        print("📊 Generating new synthetic data...")
        normal_data = generate_sensor_data(days=7, anomalous=False)
        anomalous_data = generate_sensor_data(days=7, anomalous=True)

        # Save generated data
        normal_data.to_csv('normal_week.csv', index=False)
        anomalous_data.to_csv('anomalous_week.csv', index=False)
        print("💾 Saved generated data to CSV files")

    print(f"✅ Data loaded: Normal ({len(normal_data)} records), Anomalous ({len(anomalous_data)} records)")
    return normal_data, anomalous_data

def train_autoencoder(normal_data, timesteps=24, latent_dim=32, epochs=50):
    """
    Train LSTM Autoencoder on normal behavioral data.

    Parameters:
    -----------
    normal_data : pandas.DataFrame
        Normal behavior sensor data
    timesteps : int, default=24
        Length of input sequences (hours)
    latent_dim : int, default=32  
        Dimensionality of encoded representation
    epochs : int, default=50
        Number of training epochs

    Returns:
    --------
    tuple
        (model, scaler, threshold) trained components
    """

    print("🚀 Starting LSTM Autoencoder Training")
    print("=" * 50)

    # Prepare sequences from normal data
    sequences, scaler = prepare_sequences(normal_data, timesteps=timesteps)

    # Create model architecture
    model = create_lstm_autoencoder(
        timesteps=timesteps, 
        features=sequences.shape[2], 
        latent_dim=latent_dim
    )

    print_model_summary(model)

    # Train the model on normal data only
    print(f"🎯 Training on {len(sequences)} sequences for {epochs} epochs...")

    history = model.fit(
        sequences, 
        sequences,  # Autoencoder tries to reconstruct input
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    # Calculate reconstruction errors on training data
    train_errors = calculate_reconstruction_error(model, sequences)

    # Determine anomaly detection threshold
    threshold = determine_threshold(train_errors, percentile=95)

    # Save trained model and threshold
    save_model_and_threshold(model, threshold)

    print("✅ Training completed successfully!")

    return model, scaler, threshold, history

def evaluate_anomaly_detection(model, scaler, threshold, normal_data, anomalous_data, timesteps=24):
    """
    Evaluate anomaly detection performance on test data.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained autoencoder model
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted data scaler
    threshold : float
        Anomaly detection threshold
    normal_data : pandas.DataFrame
        Normal behavior test data
    anomalous_data : pandas.DataFrame
        Anomalous behavior test data
    timesteps : int, default=24
        Sequence length

    Returns:
    --------
    dict
        Evaluation results and metrics
    """

    print("🔍 Evaluating Anomaly Detection Performance")
    print("=" * 50)

    # Prepare test sequences
    normal_sequences, _ = prepare_sequences(normal_data, timesteps=timesteps)
    anomalous_sequences, _ = prepare_sequences(anomalous_data, timesteps=timesteps)

    # Re-scale using the training scaler
    normal_features = normal_data.drop('timestamp', axis=1).values
    anomalous_features = anomalous_data.drop('timestamp', axis=1).values

    normal_scaled = scaler.transform(normal_features)
    anomalous_scaled = scaler.transform(anomalous_features)

    # Recreate sequences with proper scaling
    normal_sequences = []
    for i in range(len(normal_scaled) - timesteps + 1):
        normal_sequences.append(normal_scaled[i:i + timesteps])
    normal_sequences = np.array(normal_sequences)

    anomalous_sequences = []
    for i in range(len(anomalous_scaled) - timesteps + 1):
        anomalous_sequences.append(anomalous_scaled[i:i + timesteps])
    anomalous_sequences = np.array(anomalous_sequences)

    # Detect anomalies in normal data (should be mostly false)
    normal_anomalies, normal_errors = detect_anomalies(model, normal_sequences, threshold)

    # Detect anomalies in anomalous data (should be mostly true)
    anomalous_anomalies, anomalous_errors = detect_anomalies(model, anomalous_sequences, threshold)

    # Create ground truth labels
    y_true = np.concatenate([
        np.zeros(len(normal_sequences)),  # Normal = 0
        np.ones(len(anomalous_sequences))  # Anomalous = 1
    ])

    # Create predictions
    y_pred = np.concatenate([
        normal_anomalies.astype(int),
        anomalous_anomalies.astype(int)
    ])

    # Calculate performance metrics
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomalous']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("📊 Confusion Matrix:")
    print(cm)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'normal_errors': normal_errors,
        'anomalous_errors': anomalous_errors,
        'threshold': threshold
    }

    print(f"📈 Performance Summary:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")

    return results

def plot_reconstruction_errors(results, save_plot=False):
    """
    Plot reconstruction error distributions for normal vs anomalous data.

    Parameters:
    -----------
    results : dict
        Evaluation results from evaluate_anomaly_detection
    save_plot : bool, default=False
        Whether to save the plot
    """

    plt.figure(figsize=(12, 8))

    # Plot 1: Error distributions
    plt.subplot(2, 2, 1)
    plt.hist(results['normal_errors'], bins=30, alpha=0.7, label='Normal', color='green')
    plt.hist(results['anomalous_errors'], bins=30, alpha=0.7, label='Anomalous', color='red')
    plt.axvline(results['threshold'], color='black', linestyle='--', label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()

    # Plot 2: Time series of errors
    plt.subplot(2, 2, 2)
    plt.plot(results['normal_errors'], label='Normal', color='green', alpha=0.7)
    plt.plot(range(len(results['normal_errors']), 
                  len(results['normal_errors']) + len(results['anomalous_errors'])), 
             results['anomalous_errors'], label='Anomalous', color='red', alpha=0.7)
    plt.axhline(results['threshold'], color='black', linestyle='--', label='Threshold')
    plt.xlabel('Time Window')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error Over Time')
    plt.legend()

    # Plot 3: Confusion matrix heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=['Normal', 'Anomalous'],
                yticklabels=['Normal', 'Anomalous'],
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Plot 4: Performance metrics
    plt.subplot(2, 2, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
    bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.ylabel('Score')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_plot:
        plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')

    plt.show()

def main():
    """
    Main training and evaluation pipeline.
    """

    print("🧠 NeuroFlow: LSTM Autoencoder Training & Detection")
    print("=" * 60)

    # Step 1: Load or generate data
    normal_data, anomalous_data = load_or_generate_data()

    # Step 2: Train autoencoder on normal data
    model, scaler, threshold, history = train_autoencoder(
        normal_data, 
        timesteps=24, 
        latent_dim=32, 
        epochs=50
    )

    # Step 3: Evaluate anomaly detection performance
    results = evaluate_anomaly_detection(
        model, scaler, threshold, 
        normal_data, anomalous_data
    )

    # Step 4: Visualize results
    try:
        plot_reconstruction_errors(results, save_plot=True)
    except Exception as e:
        print(f"Note: Plotting skipped due to: {e}")

    print("🎉 NeuroFlow training and evaluation complete!")
    print("   Model and threshold saved for dashboard use.")

    return model, scaler, threshold, results

if __name__ == "__main__":
    # Run the complete training and evaluation pipeline
    model, scaler, threshold, results = main()
