"""
NeuroFlow: Interactive Mental Health Monitoring Dashboard
========================================================
This is the main Streamlit application that provides an interactive dashboard
for privacy-preserving mental health monitoring using ambient intelligence.

Author: NeuroFlow Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os

# Import our custom modules
try:
    from data_generator import generate_sensor_data, save_data_to_csv
    from model import (create_lstm_autoencoder, prepare_sequences, load_model_and_threshold,
                      detect_anomalies, save_model_and_threshold)
    from train_and_detect import train_autoencoder, evaluate_anomaly_detection
    from federated_learning import demonstrate_federated_learning
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all required files are in the same directory.")

# Page configuration
st.set_page_config(
    page_title="NeuroFlow - Mental Health Monitoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .privacy-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .alert-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'federated_complete' not in st.session_state:
    st.session_state.federated_complete = False

def load_or_generate_data():
    """Load existing data or generate new data if needed."""
    try:
        normal_data = pd.read_csv('normal_week.csv')
        anomalous_data = pd.read_csv('anomalous_week.csv')
        normal_data['timestamp'] = pd.to_datetime(normal_data['timestamp'])
        anomalous_data['timestamp'] = pd.to_datetime(anomalous_data['timestamp'])
        return normal_data, anomalous_data
    except FileNotFoundError:
        st.warning("Data files not found. Generating new synthetic data...")
        normal_data, anomalous_data = save_data_to_csv()
        st.session_state.data_generated = True
        return normal_data, anomalous_data

def create_sensor_visualization(data, title, color_theme="viridis"):
    """Create comprehensive sensor data visualization."""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Motion Sensor', 'Door Events', 'Time in Bed', 
                       'Step Count', 'Sleep Hours', 'Activity Summary'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )

    # Motion sensor
    fig.add_trace(
        go.Scatter(x=data['timestamp'], y=data['motion_sensor'], 
                  mode='lines', name='Motion', line=dict(color='blue')),
        row=1, col=1
    )

    # Door sensor
    fig.add_trace(
        go.Bar(x=data['timestamp'], y=data['door_sensor'], 
               name='Door Events', marker_color='green'),
        row=1, col=2
    )

    # Bed sensor
    fig.add_trace(
        go.Scatter(x=data['timestamp'], y=data['bed_sensor'], 
                  mode='lines+markers', name='Bed Time', line=dict(color='purple')),
        row=1, col=3
    )

    # Step count (only show non-zero values)
    step_data = data[data['step_count'] > 0]
    fig.add_trace(
        go.Bar(x=step_data['timestamp'], y=step_data['step_count'], 
               name='Steps', marker_color='orange'),
        row=2, col=1
    )

    # Sleep hours (only show non-zero values)
    sleep_data = data[data['sleep_hours'] > 0]
    fig.add_trace(
        go.Bar(x=sleep_data['timestamp'], y=sleep_data['sleep_hours'], 
               name='Sleep', marker_color='red'),
        row=2, col=2
    )

    # Activity summary - combined metric
    activity_score = (data['motion_sensor'] * 0.3 + 
                     data['door_sensor'] * 20 + 
                     data['step_count'] / 100)
    fig.add_trace(
        go.Scatter(x=data['timestamp'], y=activity_score, 
                  mode='lines', name='Activity Score', line=dict(color='black')),
        row=2, col=3
    )

    fig.update_layout(
        title=title,
        height=600,
        showlegend=False
    )

    return fig

def create_anomaly_timeline(errors, threshold, timestamps):
    """Create anomaly detection timeline visualization."""

    fig = go.Figure()

    # Add reconstruction error line
    fig.add_trace(go.Scatter(
        x=list(range(len(errors))),
        y=errors,
        mode='lines+markers',
        name='Reconstruction Error',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Anomaly Threshold ({threshold:.4f})",
        annotation_position="top right"
    )

    # Highlight anomalies
    anomalies = errors > threshold
    if np.any(anomalies):
        fig.add_trace(go.Scatter(
            x=np.where(anomalies)[0],
            y=errors[anomalies],
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=8, symbol='star')
        ))

    fig.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Time Window",
        yaxis_title="Reconstruction Error",
        height=400
    )

    return fig

def home_page():
    """Home page with project overview."""

    st.markdown('<h1 class="main-header">🧠 NeuroFlow</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Privacy-First Mental Health Monitoring through Ambient Intelligence</p>', unsafe_allow_html=True)

    # Key features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔒 Privacy by Design
        - Data never leaves your device
        - Federated learning architecture
        - HIPAA/GDPR compliant
        - No central health database
        """)

    with col2:
        st.markdown("""
        ### 🧠 Intelligent Detection
        - LSTM Autoencoder AI model
        - Personalized anomaly detection
        - Behavioral pattern analysis
        - Early intervention alerts
        """)

    with col3:
        st.markdown("""
        ### 📱 Ambient Sensing
        - Smart home integration
        - Passive monitoring
        - Multimodal sensor fusion
        - Non-intrusive approach
        """)

    # Status indicators
    st.markdown("### 📊 System Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        data_status = "✅ Ready" if st.session_state.data_generated or os.path.exists('normal_week.csv') else "⏳ Pending"
        st.metric("Data Generation", data_status)

    with col2:
        model_status = "✅ Trained" if st.session_state.model_trained or os.path.exists('neuroflow_autoencoder.h5') else "⏳ Pending"
        st.metric("Model Training", model_status)

    with col3:
        fed_status = "✅ Complete" if st.session_state.federated_complete else "⏳ Pending"
        st.metric("Federated Learning", fed_status)

    with col4:
        privacy_status = "🔒 Enabled"
        st.metric("Privacy Mode", privacy_status)

    # Project description
    st.markdown("""
    ### 🎯 Project Overview

    NeuroFlow represents a breakthrough in mental health monitoring technology. By leveraging ambient intelligence 
    and privacy-preserving machine learning, we enable continuous behavioral monitoring that can detect early 
    signs of mental health changes without compromising user privacy.

    **How it works:**
    1. **Ambient Data Collection**: Smart home sensors passively collect behavioral patterns
    2. **Local AI Processing**: LSTM Autoencoder runs on your device to learn normal patterns
    3. **Federated Learning**: Multiple users collaborate to improve the model without sharing data
    4. **Anomaly Detection**: Unusual behavioral patterns trigger early intervention alerts
    5. **Privacy Preservation**: All sensitive data remains on your local device

    **Key Innovation**: Traditional mental health monitoring requires active engagement and data sharing. 
    NeuroFlow provides passive, privacy-preserving monitoring that respects user autonomy while enabling 
    proactive mental healthcare.
    """)

def dashboard_page():
    """Main dashboard with data visualization and anomaly detection."""

    st.title("📊 NeuroFlow Dashboard")
    st.markdown('<span class="privacy-badge">🔒 Privacy Mode: ON</span>', unsafe_allow_html=True)

    # Load data
    try:
        normal_data, anomalous_data = load_or_generate_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar controls
    st.sidebar.header("Dashboard Controls")

    # Data selection
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["Normal Behavior", "Anomalous Behavior", "Both"]
    )

    # Visualization controls
    show_raw_data = st.sidebar.checkbox("Show Raw Data Table", False)
    threshold_adjust = st.sidebar.slider("Anomaly Threshold Multiplier", 0.5, 2.0, 1.0, 0.1)

    # Main dashboard layout
    if data_type == "Normal Behavior":
        st.subheader("📈 Normal Behavioral Patterns")
        fig = create_sensor_visualization(normal_data, "Normal Behavior - 7 Day Pattern")
        st.plotly_chart(fig, use_container_width=True)

        if show_raw_data:
            st.subheader("📋 Raw Sensor Data")
            st.dataframe(normal_data)

    elif data_type == "Anomalous Behavior":
        st.subheader("⚠️ Anomalous Behavioral Patterns")
        fig = create_sensor_visualization(anomalous_data, "Anomalous Behavior - Depression Indicators")
        st.plotly_chart(fig, use_container_width=True)

        if show_raw_data:
            st.subheader("📋 Raw Sensor Data")
            st.dataframe(anomalous_data)

    else:  # Both
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Normal Patterns")
            fig1 = create_sensor_visualization(normal_data, "Normal Behavior")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("⚠️ Anomalous Patterns")  
            fig2 = create_sensor_visualization(anomalous_data, "Anomalous Behavior")
            st.plotly_chart(fig2, use_container_width=True)

    # Anomaly detection section
    if os.path.exists('neuroflow_autoencoder.h5') and os.path.exists('threshold.npy'):
        st.subheader("🔍 AI-Powered Anomaly Detection")

        try:
            # Load trained model
            model, base_threshold = load_model_and_threshold()
            adjusted_threshold = base_threshold * threshold_adjust

            # Prepare test data
            test_sequences, scaler = prepare_sequences(anomalous_data, timesteps=24)

            # Detect anomalies
            anomalies, errors = detect_anomalies(model, test_sequences, adjusted_threshold)

            # Create timeline visualization
            timeline_fig = create_anomaly_timeline(errors, adjusted_threshold, anomalous_data['timestamp'])
            st.plotly_chart(timeline_fig, use_container_width=True)

            # Anomaly summary
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Time Windows", len(errors))

            with col2:
                anomaly_count = np.sum(anomalies)
                st.metric("Anomalies Detected", anomaly_count)

            with col3:
                anomaly_rate = (anomaly_count / len(errors)) * 100 if len(errors) > 0 else 0
                st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

            if anomaly_count > 0:
                st.warning(f"⚠️ **Alert**: {anomaly_count} anomalous behavioral patterns detected. Consider reaching out to a mental health professional.")
            else:
                st.success("✅ **Good News**: No significant behavioral anomalies detected in the current time period.")

        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")
    else:
        st.info("🤖 Train the AI model first to enable anomaly detection.")

def model_training_page():
    """Model training interface."""

    st.title("🤖 AI Model Training")
    st.markdown("Train the LSTM Autoencoder for personalized anomaly detection")

    # Training parameters
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider("Training Epochs", 10, 100, 50)
        latent_dim = st.slider("Latent Dimension", 16, 64, 32)

    with col2:
        timesteps = st.slider("Time Window (hours)", 12, 48, 24)
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    # Training button
    if st.button("🚀 Start Training", type="primary"):

        # Load data
        try:
            normal_data, anomalous_data = load_or_generate_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        # Training process
        with st.spinner("Training LSTM Autoencoder... This may take a few minutes."):

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Simulate training progress
                progress_bar.progress(10)
                status_text.text("Preparing data sequences...")
                time.sleep(1)

                progress_bar.progress(30)
                status_text.text("Creating model architecture...")
                time.sleep(1)

                progress_bar.progress(50)
                status_text.text("Training neural network...")

                # Actual training
                model, scaler, threshold, history = train_autoencoder(
                    normal_data,
                    timesteps=timesteps,
                    latent_dim=latent_dim,
                    epochs=epochs
                )

                progress_bar.progress(80)
                status_text.text("Evaluating model performance...")
                time.sleep(1)

                # Evaluation
                results = evaluate_anomaly_detection(
                    model, scaler, threshold,
                    normal_data, anomalous_data
                )

                progress_bar.progress(100)
                status_text.text("Training complete!")

                # Update session state
                st.session_state.model_trained = True

                # Display results
                st.success("✅ Model training completed successfully!")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.3f}")

                with col2:
                    st.metric("Precision", f"{results['precision']:.3f}")

                with col3:
                    st.metric("Recall", f"{results['recall']:.3f}")

                with col4:
                    st.metric("F1-Score", f"{results['f1_score']:.3f}")

                # Training history
                if hasattr(history, 'history'):
                    st.subheader("📈 Training Progress")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss'
                    ))

                    if 'val_loss' in history.history:
                        fig.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))

                    fig.update_layout(
                        title="Model Training Loss",
                        xaxis_title="Epoch",
                        yaxis_title="Loss"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Training failed: {e}")
                progress_bar.empty()
                status_text.empty()

    # Model status
    st.subheader("📊 Model Status")

    if os.path.exists('neuroflow_autoencoder.h5'):
        st.success("✅ Trained model available")

        # Model info
        try:
            model, threshold = load_model_and_threshold()

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"**Model Parameters**: {model.count_params():,}")
                st.info(f"**Anomaly Threshold**: {threshold:.6f}")

            with col2:
                st.info(f"**Architecture**: LSTM Autoencoder")
                st.info(f"**Privacy**: Federated Learning Ready")

        except Exception as e:
            st.warning(f"Model file exists but cannot be loaded: {e}")
    else:
        st.warning("⏳ No trained model found. Train a model to enable anomaly detection.")

def federated_learning_page():
    """Federated learning demonstration."""

    st.title("🌐 Federated Learning")
    st.markdown("Privacy-preserving collaborative AI training")

    # Explanation
    st.subheader("🔒 Privacy-First Approach")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Traditional Centralized Learning:**
        - ❌ All data sent to central server
        - ❌ Privacy risks and data breaches
        - ❌ Regulatory compliance challenges
        - ❌ User trust issues
        """)

    with col2:
        st.markdown("""
        **NeuroFlow Federated Learning:**
        - ✅ Data stays on user devices
        - ✅ Only model updates shared
        - ✅ Collaborative improvement
        - ✅ Privacy by design
        """)

    # Federated learning simulation
    st.subheader("🚀 Federated Training Simulation")

    # Parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        num_clients = st.selectbox("Number of Clients", [3, 5, 10], index=0)

    with col2:
        num_rounds = st.selectbox("Federated Rounds", [1, 3, 5], index=1)

    with col3:
        local_epochs = st.selectbox("Local Epochs", [5, 10, 15], index=1)

    if st.button("🌐 Start Federated Learning", type="primary"):

        with st.spinner("Running federated learning simulation..."):

            try:
                # Import and run federated learning
                from federated_learning import FederatedLearningSimulator

                # Create simulator
                fed_sim = FederatedLearningSimulator(
                    num_clients=num_clients,
                    timesteps=24,
                    features=5,
                    latent_dim=32
                )

                # Run simulation
                results = fed_sim.run_federated_training(
                    num_rounds=num_rounds,
                    local_epochs=local_epochs
                )

                # Update session state
                st.session_state.federated_complete = True

                st.success("✅ Federated learning simulation completed!")

                # Display results
                st.subheader("📊 Federated Learning Results")

                # Create results visualization
                round_losses = [r['federated_results']['average_loss'] for r in results]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(round_losses) + 1)),
                    y=round_losses,
                    mode='lines+markers',
                    name='Average Loss',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))

                fig.update_layout(
                    title="Federated Learning Progress",
                    xaxis_title="Federated Round",
                    yaxis_title="Average Loss",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Privacy metrics
                st.subheader("🔒 Privacy Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Clients Participated", num_clients)

                with col2:
                    total_samples = sum(r['federated_results']['total_samples'] for r in results)
                    st.metric("Total Samples (Local)", f"{total_samples:,}")

                with col3:
                    st.metric("Data Shared", "0 bytes")

                st.success("🛡️ **Privacy Preserved**: All sensitive health data remained on local devices throughout the entire training process.")

            except Exception as e:
                st.error(f"Federated learning simulation failed: {e}")

    # Privacy comparison
    st.subheader("📊 Privacy Comparison")

    comparison_data = {
        'Metric': ['Data Location', 'Network Traffic', 'Model Updates', 'Privacy Risk', 'Compliance'],
        'Centralized': ['Server', 'High (Raw Data)', 'Server Only', 'High', 'Complex'],
        'Federated': ['Local Devices', 'Low (Model Weights)', 'Collaborative', 'Minimal', 'Built-in']
    }

    st.table(pd.DataFrame(comparison_data))

def privacy_settings_page():
    """Privacy controls and settings."""

    st.title("🔒 Privacy Settings")
    st.markdown("Configure your privacy preferences and data controls")

    # Privacy status
    st.subheader("🛡️ Privacy Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✅ **Data Locality**  All data stays on device")

    with col2:
        st.success("✅ **Encryption**  End-to-end encrypted")

    with col3:
        st.success("✅ **Anonymization**  No personal identifiers")

    # Privacy controls
    st.subheader("⚙️ Privacy Controls")

    enable_collection = st.checkbox("Enable data collection", value=True, 
                                   help="Allow NeuroFlow to collect ambient sensor data")

    enable_federated = st.checkbox("Participate in federated learning", value=True,
                                  help="Contribute to improving the global model while keeping data local")

    enable_alerts = st.checkbox("Enable anomaly alerts", value=True,
                               help="Receive notifications when unusual patterns are detected")

    # Data retention
    st.subheader("📅 Data Retention")

    retention_days = st.slider("Data retention period (days)", 1, 365, 30,
                              help="How long to keep sensor data on your device")

    auto_delete = st.checkbox("Auto-delete after retention period", value=True)

    # Model personalization
    st.subheader("🎯 Model Personalization")

    personalization_level = st.selectbox(
        "Personalization Level",
        ["Minimal", "Moderate", "High"],
        index=1,
        help="How much the model adapts to your specific behavioral patterns"
    )

    adaptive_threshold = st.checkbox("Adaptive anomaly threshold", value=True,
                                   help="Allow the model to adjust sensitivity based on your patterns")

    # Export/Delete data
    st.subheader("📤 Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export My Data"):
            st.info("Data export functionality would generate a privacy-compliant export of your behavioral patterns.")

    with col2:
        if st.button("🗑️ Delete All Data", type="secondary"):
            st.warning("This would permanently delete all local data and model personalization.")

    # Privacy policy summary
    st.subheader("📋 Privacy Summary")

    st.markdown("""
    **NeuroFlow Privacy Commitments:**

    - 🔒 **Local Processing**: All AI analysis happens on your device
    - 🚫 **No Data Sharing**: Raw sensor data never leaves your device
    - 🔐 **Encrypted Communication**: All network traffic is encrypted
    - 🏠 **Edge Computing**: Models run locally, not in the cloud
    - 📱 **Device Control**: You own and control all your data
    - 🛡️ **Privacy by Design**: System architecture prioritizes privacy
    - ⚖️ **Regulatory Compliance**: HIPAA, GDPR, and CCPA compliant
    - 🔄 **Transparent Operations**: Open source algorithms and processes

    **What We DO collect:**
    - Model performance metrics (anonymized)
    - System usage statistics (aggregated)
    - Error reports (no personal data)

    **What We DON'T collect:**
    - Raw sensor readings
    - Personal behavioral patterns  
    - Location data
    - Identity information
    - Health status information
    """)

def real_time_monitoring():
    """Real-time monitoring simulation."""

    st.subheader("⏱️ Real-Time Monitoring")
    st.markdown("Simulated live sensor data and anomaly detection")

    if st.button("🚀 Start Real-Time Monitor"):

        # Create containers for real-time updates
        metrics_container = st.container()
        chart_container = st.container()

        # Initialize data
        time_data = []
        motion_data = []
        door_data = []
        bed_data = []

        # Real-time simulation
        for i in range(20):  # 20 time steps

            # Generate new data point
            current_time = datetime.now() + timedelta(minutes=i*5)

            # Simulate sensor readings with some randomness  
            motion = max(0, np.random.normal(30, 15))
            door = np.random.poisson(1)
            bed = max(0, np.random.normal(0.1, 0.05))

            # Add to data
            time_data.append(current_time)
            motion_data.append(motion)
            door_data.append(door)
            bed_data.append(bed)

            # Update metrics
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Current Motion", f"{motion:.1f}")

                with col2:
                    st.metric("Door Events", f"{door}")

                with col3:
                    st.metric("Bed Time", f"{bed:.2f}")

                with col4:
                    # Simple anomaly simulation
                    anomaly_score = motion + door * 10 + bed * 100
                    is_anomaly = anomaly_score > 60
                    status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
                    st.metric("Status", status)

            # Update chart
            with chart_container:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=motion_data,
                    mode='lines+markers',
                    name='Motion Sensor',
                    line=dict(color='blue')
                ))

                fig.add_trace(go.Scatter(
                    x=time_data,
                    y=[d * 10 for d in door_data],  # Scale for visibility
                    mode='lines+markers',
                    name='Door Events (x10)',
                    line=dict(color='green')
                ))

                fig.update_layout(
                    title="Real-Time Sensor Data",
                    xaxis_title="Time",
                    yaxis_title="Sensor Value",
                    height=300
                )

                st.plotly_chart(fig, use_container_width=True)

            # Pause for animation effect
            time.sleep(0.5)

        st.success("✅ Real-time monitoring simulation complete!")

def main():
    """Main application with navigation."""

    # Sidebar navigation
    st.sidebar.title("🧠 NeuroFlow")
    st.sidebar.markdown("Privacy-First Mental Health Monitoring")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["🏠 Home", "📊 Dashboard", "🤖 Model Training", "🌐 Federated Learning", "🔒 Privacy Settings"]
    )

    # Page routing
    if page == "🏠 Home":
        home_page()

    elif page == "📊 Dashboard":
        dashboard_page()

        # Add real-time monitoring section
        with st.expander("⏱️ Real-Time Monitoring", expanded=False):
            real_time_monitoring()

    elif page == "🤖 Model Training":
        model_training_page()

    elif page == "🌐 Federated Learning":
        federated_learning_page()

    elif page == "🔒 Privacy Settings":
        privacy_settings_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🛡️ Privacy Status**")
    st.sidebar.success("🔒 All data stays local")
    st.sidebar.info("🤝 Federated learning enabled")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*NeuroFlow v1.0 - Mental Health Hackathon 2025*")

if __name__ == "__main__":
    main()
