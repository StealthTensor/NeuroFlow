#!/usr/bin/env python3
"""
NeuroFlow Test Suite
====================
Quick test script to verify all components are working correctly.

Run this script to ensure your NeuroFlow installation is complete and functional.
"""

import sys
import os
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print(" Testing imports...")

    try:
        import tensorflow as tf
        print(f"    TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"    TensorFlow import failed: {e}")
        return False

    try:
        import streamlit as st
        print(f"    Streamlit")
    except ImportError as e:
        print(f"    Streamlit import failed: {e}")
        return False

    try:
        import plotly
        print(f"    Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"    Plotly import failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"    Pandas {pd.__version__}")
    except ImportError as e:
        print(f"    Pandas import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"    NumPy {np.__version__}")
    except ImportError as e:
        print(f"    NumPy import failed: {e}")
        return False

    try:
        import sklearn
        print(f"    Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"    Scikit-learn import failed: {e}")
        return False

    return True

def test_custom_modules():
    """Test that all custom NeuroFlow modules can be imported."""
    print(" Testing NeuroFlow modules...")

    required_files = [
        'data_generator.py',
        'model.py', 
        'train_and_detect.py',
        'federated_learning.py',
        'neuroflow_app.py'
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"    {file} found")
        else:
            print(f"    {file} missing")
            return False

    try:
        from data_generator import generate_sensor_data
        print("    data_generator module imported")
    except ImportError as e:
        print(f"    data_generator import failed: {e}")
        return False

    try:
        from model import create_lstm_autoencoder
        print("    model module imported")
    except ImportError as e:
        print(f"    model import failed: {e}")
        return False

    return True

def test_data_generation():
    """Test synthetic data generation."""
    print(" Testing data generation...")

    try:
        from data_generator import generate_sensor_data

        # Generate small test dataset
        test_data = generate_sensor_data(days=1, anomalous=False)

        if len(test_data) > 0:
            print(f"    Generated {len(test_data)} data points")
            print(f"    Data columns: {list(test_data.columns)}")
            return True
        else:
            print("    No data generated")
            return False

    except Exception as e:
        print(f"    Data generation failed: {e}")
        return False

def test_model_creation():
    """Test LSTM Autoencoder model creation."""
    print(" Testing model creation...")

    try:
        from model import create_lstm_autoencoder

        # Create small test model
        model = create_lstm_autoencoder(timesteps=10, features=5, latent_dim=16)

        if model is not None:
            print(f"    Model created with {model.count_params()} parameters")
            return True
        else:
            print("    Model creation returned None")
            return False

    except Exception as e:
        print(f"    Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_federated_learning():
    """Test federated learning simulator."""
    print(" Testing federated learning...")

    try:
        from federated_learning import FederatedLearningSimulator

        # Create small simulator
        fed_sim = FederatedLearningSimulator(num_clients=2, timesteps=10, features=5, latent_dim=16)

        if fed_sim is not None:
            print("    Federated learning simulator created")
            return True
        else:
            print("    Federated learning simulator creation failed")
            return False

    except Exception as e:
        print(f"    Federated learning test failed: {e}")
        return False

def run_full_test():
    """Run complete test suite."""
    print(" NeuroFlow Test Suite")
    print("=" * 40)

    tests = [
        ("Import Dependencies", test_imports),
        ("Custom Modules", test_custom_modules), 
        ("Data Generation", test_data_generation),
        ("Model Creation", test_model_creation),
        ("Federated Learning", test_federated_learning)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f" {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
                print(f" {test_name}: PASSED")
            else:
                print(f" {test_name}: FAILED")
        except Exception as e:
            print(f" {test_name}: ERROR - {e}")

    print(f" Test Results: {passed}/{total} tests passed")

    if passed == total:
        print(" All tests passed! NeuroFlow is ready to use.")
        print(" Next steps:")
        print("   1. Run: python data_generator.py")
        print("   2. Run: python train_and_detect.py") 
        print("   3. Run: streamlit run neuroflow_app.py")
    else:
        print("  Some tests failed. Please check the error messages above.")
        print(" Try: pip install -r requirements.txt")

    return passed == total

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
