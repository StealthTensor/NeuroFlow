"""
NeuroFlow Data Generator
========================
This module generates synthetic multimodal sensor data for mental health monitoring.
It creates realistic patterns for normal and anomalous (depressive) behaviors.

Author: NeuroFlow Team
Date: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_sensor_data(days=7, anomalous=False, start_date=None):
    """
    Generate synthetic multimodal sensor data for NeuroFlow project.

    Parameters:
    -----------
    days : int, default=7
        Number of days to generate data for
    anomalo : bool, default=False
        Whether to generate anomalous (depressive) behavior patterns
    start_date : datetime, optional
        Starting date for the data generation. If None, uses current date minus days.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing timestamp and sensor readings
    """

    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    # Generate hourly timestamps for the specified period
    timestamps = pd.date_range(
        start=start_date,
        end=start_date + timedelta(days=days),
        freq='h'
    )

    # Initialize data structure
    data = {
        'timestamp': timestamps,
        'motion_sensor': [],      # PIR motion sensor readings per hour
        'door_sensor': [],        # Number of door open events per hour
        'bed_sensor': [],         # Hours spent in bed
        'step_count': [],         # Daily step count (distributed hourly)
        'sleep_hours': []         # Sleep duration per day
    }

    # Set random seed for reproducible results
    np.random.seed(42 if not anomalous else 24)

    # Generate data for each timestamp
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()  # 0=Monday, 6=Sunday

        if anomalous:
            # Simulate depressive behavior patterns
            # Reduced activity, more time in bed, irregular sleep

            # Motion sensor: significantly reduced activity
            if 6 <= hour <= 10:  # Morning sluggishness
                motion = max(0, np.random.normal(15, 8))
            elif 11 <= hour <= 16:  # Afternoon lethargy
                motion = max(0, np.random.normal(8, 5))
            elif 17 <= hour <= 22:  # Evening minimal activity
                motion = max(0, np.random.normal(5, 3))
            else:  # Night
                motion = max(0, np.random.normal(2, 1))

            # Door sensor: fewer outings, social isolation
            if 8 <= hour <= 20 and day_of_week < 5:  # Weekdays
                door_opens = np.random.poisson(0.5)  # Much less activity
            elif 8 <= hour <= 20:  # Weekends
                door_opens = np.random.poisson(0.3)
            else:
                door_opens = 0

            # Bed sensor: excessive time in bed
            if 22 <= hour or hour <= 11:  # Extended bed time
                bed_time = max(0, np.random.normal(0.8, 0.2))  # 80% of hour in bed
            elif 12 <= hour <= 17:  # Afternoon napping
                bed_time = max(0, np.random.normal(0.3, 0.2))  # Frequent naps
            else:
                bed_time = max(0, np.random.normal(0.1, 0.1))

            # Step count: significantly reduced daily activity
            if hour == 0:  # Daily step count assigned at midnight
                steps = max(500, np.random.normal(2000, 800))  # Very low activity
            else:
                steps = 0  # Steps only counted once per day

            # Sleep hours: irregular and often excessive
            if hour == 0:  # Sleep duration assigned at midnight
                sleep = max(4, min(14, np.random.normal(10, 2)))  # 8-12 hours, often excessive
            else:
                sleep = 0

        else:
            # Normal behavior patterns
            # Regular activity cycles, healthy sleep patterns

            # Motion sensor: healthy activity patterns
            if 6 <= hour <= 9:  # Morning routine
                motion = max(0, np.random.normal(45, 15))
            elif 10 <= hour <= 17:  # Daytime activity
                motion = max(0, np.random.normal(35, 12))
            elif 18 <= hour <= 22:  # Evening activity
                motion = max(0, np.random.normal(40, 10))
            else:  # Night rest
                motion = max(0, np.random.normal(3, 2))

            # Door sensor: regular outings and social activity
            if 8 <= hour <= 20 and day_of_week < 5:  # Weekdays
                door_opens = np.random.poisson(2)  # Regular activity
            elif 8 <= hour <= 20:  # Weekends
                door_opens = np.random.poisson(3)  # More weekend activity
            else:
                door_opens = 0

            # Bed sensor: normal sleep schedule
            if 22 <= hour or hour <= 7:  # Normal bedtime
                bed_time = max(0, np.random.normal(0.7, 0.1))
            else:
                bed_time = max(0, np.random.normal(0.05, 0.05))  # Minimal daytime bed use

            # Step count: healthy daily activity
            if hour == 0:  # Daily step count assigned at midnight
                steps = max(2000, np.random.normal(8000, 2000))  # Healthy activity level
            else:
                steps = 0

            # Sleep hours: regular sleep pattern
            if hour == 0:  # Sleep duration assigned at midnight
                sleep = max(6, min(9, np.random.normal(7.5, 0.8)))  # 7-8 hours typically
            else:
                sleep = 0

        # Append generated values to data structure
        data['motion_sensor'].append(motion)
        data['door_sensor'].append(door_opens)
        data['bed_sensor'].append(bed_time)
        data['step_count'].append(steps)
        data['sleep_hours'].append(sleep)

    return pd.DataFrame(data)

def plot_sensor_data(normal_data, anomalous_data, save_plot=False):
    """
    Plot comparison between normal and anomalous sensor data.

    Parameters:
    -----------
    normal_data : pandas.DataFrame
        Normal behavior data
    anomalous_data : pandas.DataFrame
        Anomalous behavior data
    save_plot : bool, default=False
        Whether to save the plot as PNG file
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('NeuroFlow: Normal vs Anomalous Behavior Patterns', fontsize=16)

    sensors = ['motion_sensor', 'door_sensor', 'bed_sensor', 'step_count', 'sleep_hours']
    titles = ['Motion Sensor Activity', 'Door Opening Events', 'Time in Bed',
             'Daily Step Count', 'Sleep Hours']

    for i, (sensor, title) in enumerate(zip(sensors, titles)):
        row = i // 3
        col = i % 3

        if i < 5:  # Only plot first 5 sensors
            axes[row, col].plot(normal_data['timestamp'], normal_data[sensor],
                              label='Normal', alpha=0.7, color='green')
            axes[row, col].plot(anomalous_data['timestamp'], anomalous_data[sensor],
                              label='Anomalous', alpha=0.7, color='red')
            axes[row, col].set_title(title)
            axes[row, col].legend()
            axes[row, col].tick_params(axis='x', rotation=45)

    # Remove the empty subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()

    if save_plot:
        plt.savefig('sensor_data_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()

def save_data_to_csv():
    """
    Generate and save both normal and anomalous datasets to CSV files.
    """

    print("🧠 NeuroFlow Data Generator")
    print("=" * 40)
    print("Generating synthetic sensor data...")

    # Generate normal behavior data
    print("📊 Generating normal behavior patterns...")
    normal_data = generate_sensor_data(days=7, anomalous=False)

    # Generate anomalous behavior data
    print("⚠️  Generating anomalous behavior patterns...")
    anomalous_data = generate_sensor_data(days=7, anomalous=True)

    # Save to CSV files
    normal_data.to_csv('normal_week.csv', index=False)
    anomalous_data.to_csv('anomalous_week.csv', index=False)

    print("✅ Data generation complete!")
    print(f"   📄 Normal data saved to: normal_week.csv ({len(normal_data)} records)")
    print(f"   📄 Anomalous data saved to: anomalous_week.csv ({len(anomalous_data)} records)")

    # Display summary statistics
    print("\n📈 Data Summary:")
    print("Normal Data Statistics:")
    print(normal_data.describe())
    print("\nAnomalous Data Statistics:")
    print(anomalous_data.describe())

    return normal_data, anomalous_data

if __name__ == "__main__":
    # Generate and save datasets
    normal_data, anomalous_data = save_data_to_csv()

    # Optionally plot the data comparison
    try:
        plot_sensor_data(normal_data, anomalous_data, save_plot=True)
    except Exception as e:
        print(f"Note: Plotting skipped due to: {e}")