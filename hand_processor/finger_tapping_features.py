import os
import sys
# Adjust the working directory to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)
os.chdir(project_root)
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from pycatch22 import catch22_all
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema
from config import BASE_PROCESSED_DIRECTORY

FINGER_DISTANCE = 'Finger Distance'
FINGER_NORMALIZED_DISTANCE = 'Finger Normalized Distance'
ANGULAR_DISTANCE = 'Angular Distance'
WRIST_COORDINATE = 'Wrist Coordinate'
FRAME = 'Frame'

class FeatureExtractor:
    def __init__(self, base_processed_directory=None, fps=60):
        self.base_processed_directory = base_processed_directory
        self.fps = fps

    def calculate_wrist_mobility(self, wrist_coordinates):
        wrist_coordinates = np.array(wrist_coordinates)
        wrist_coordinates = [eval(i) for i in wrist_coordinates if i is not np.nan]
        distances = np.sqrt(np.sum(np.diff(wrist_coordinates, axis=0) ** 2, axis=1))
        distances = distances[distances <= 50]
        total_distance = np.sum(distances)
        mean_speed = np.mean(distances)
        std_speed = np.std(distances)
        return total_distance, mean_speed, std_speed

    def find_peaks_finger_tapping(self, data):
        """
        Find minima (tap indices) and maxima (max distances between taps) in the FINGER_NORMALIZED_DISTANCE data.
        Returns maxima indices, minima indices, start index, end index.
        """
        distances = data[FINGER_NORMALIZED_DISTANCE].values

        # Find taps in the data (minima)
        tap_count, taps_indices = self.count_taps(distances)

        # Find the best segment based on taps
        start_index, end_index = self.find_best_segment_indices(distances, taps_indices)

        # Trim distances and taps indices to the best segment
        distances_segment = distances[start_index:end_index + 1]
        taps_indices = taps_indices[(taps_indices >= start_index) & (taps_indices <= end_index)] - start_index

        # Find maxima between taps
        maxima_indices = []
        for i in range(len(taps_indices) - 1):
            start_tap = taps_indices[i]
            end_tap = taps_indices[i + 1]
            # Find maximum between two taps
            if end_tap > start_tap + 1:
                local_max_index = np.argmax(distances_segment[start_tap:end_tap + 1]) + start_tap
                maxima_indices.append(local_max_index + start_index)  # Adjust back to original indices

        # Adjust minima indices (taps_indices) back to original indices
        minima_indices = taps_indices + start_index

        return distances ,maxima_indices, minima_indices, start_index, end_index
    
    def count_taps(self, distances, min_prominence=0.15, distance = 10, height_multiplyer = 0.3):
        distances = np.array(distances, dtype=float)
        taps_indices = argrelextrema(distances, np.less)[0]
        peaks, _ = find_peaks(-distances, prominence=min_prominence, distance=distance, height=-height_multiplyer*(max(distances)))
        taps_indices = np.intersect1d(taps_indices, peaks)
        tap_count = len(taps_indices)
        return tap_count, taps_indices
    
    def calculate_amplitudes(self,distances,taps_indices):
        amplitudes = []
        max_amplitude_positions = []
        for i, index in enumerate(taps_indices):
            if i == len(taps_indices) - 1:
                break
            start = max(0, index)
            end = min(len(distances), taps_indices[i + 1])
            local_max = np.max(distances[start:end])
            amplitude = local_max
            amplitudes.append(amplitude)
            max_amplitude_positions.append(np.argmax(distances[start:end]) + start)
        return max_amplitude_positions, amplitudes

    def calculate_fft_features(self, distances):
            try:
                n = len(distances)
                T = 1.0 / self.fps
                yf = fft(np.array(distances))
                xf = fftfreq(n, T)[:n // 2]

                yf_abs = 2.0 / n * np.abs(yf[:n // 2])
                xf_nonzero = xf[1:]
                yf_abs_nonzero = yf_abs[1:]

                peak_index = np.argmax(yf_abs_nonzero)
                peak_frequency = xf_nonzero[peak_index]
                peak_amplitude = yf_abs_nonzero[peak_index]
                num_peaks, _ = find_peaks(yf_abs_nonzero, height=20)

                power = np.sum(yf_abs_nonzero ** 2)
                frequency_consistency = np.std(xf_nonzero)

                return peak_frequency, peak_amplitude, power, len(num_peaks), frequency_consistency
            except Exception as e:
                print(f"Error calculating FFT features: {e}")
                return 0, 0, 0, 0, 0
    
    def find_best_segment_indices(self, distances, taps_indices, cutoff=1.0, fs=10.0, order=5, min_prominence=0.15, distance=10, height_multiplier=0.3, min_taps=8):
        

        def split_segments(distances, taps_indices):
            max_gap = np.mean(np.diff(taps_indices)) * 3
            segments = []
            start_idx = taps_indices[0]
            for i in range(1, len(taps_indices)):
                if taps_indices[i] - taps_indices[i - 1] > max_gap:
                    segments.append((start_idx, taps_indices[i - 1]))
                    start_idx = taps_indices[i]
            segments.append((start_idx, taps_indices[-1]))
            return segments

        def select_best_segment(distances, segments):
            best_segment = None
            max_taps = 0
            for start, end in segments:
                segment_distances = distances[start:end + 1]
                tap_count, _ = self.count_taps(segment_distances, min_prominence, distance, height_multiplier)
                
                if tap_count > max_taps and tap_count > min_taps:
                    max_taps = tap_count
                    best_segment = (start, end)
            if best_segment is None:
                best_segment = (segments[0][0], segments[-1][1])
            return best_segment
        
        
        # Split the data into segments based on the gap between taps
        segments = split_segments(distances, taps_indices)

        # Select the best segment with the highest number of taps
        best_segment = select_best_segment(distances, segments)

        # Return the start and last index of the best segment
        return best_segment

    def calculate_tapping_frequency(self, tap_count, duration):
        return tap_count / duration if duration > 0 else 0
    
    def calculate_decrements(self, amplitudes, taps_indices, fps):
        if len(amplitudes) < 6:
            return 0, 0, 0, 0
        
        mean_first_3_amplitudes = np.mean(amplitudes[:3])
        mean_last_3_amplitudes = np.mean(amplitudes[-3:])
        amplitude_decrement = mean_first_3_amplitudes - mean_last_3_amplitudes

        tap_intervals = np.diff(taps_indices) / fps
        if len(tap_intervals) < 6:
            return amplitude_decrement, 0, 0, 0

        mean_first_3_intervals = np.mean(tap_intervals[:3])
        mean_last_3_intervals = np.mean(tap_intervals[-3:])
        frequency_decrement = (1 / mean_first_3_intervals) - (1 / mean_last_3_intervals)
        
        tap_speeds = 1 / tap_intervals
        halt_threshold = 2 * np.mean(tap_intervals)
        halts_and_hesitations = np.sum(tap_intervals > halt_threshold)
        
        return amplitude_decrement, frequency_decrement, np.mean(tap_speeds), halts_and_hesitations
    
    def extract_features_from_distances(self, distances, output_path=None):
        """
        Extract all features (tapping, amplitudes, FFT, wrist mobility) from the distance data.
        """
        if len(distances) <5:
            return None
        # Extract best segment indices for trimming
        tap_count, taps_indices = self.count_taps(distances[FINGER_NORMALIZED_DISTANCE])
        start_index, last_index = self.find_best_segment_indices(distances[FINGER_NORMALIZED_DISTANCE], taps_indices)

        # Trim distances and pose data
        trimmed_distances = distances[FINGER_NORMALIZED_DISTANCE][start_index:last_index + 1]
        trimmed_pose_data = distances[WRIST_COORDINATE][start_index:last_index + 1]

        # Extract wrist mobility
        total_distance, mean_speed, std_speed = self.calculate_wrist_mobility(trimmed_pose_data)

        # Extract tapping and amplitude features
        trimmed_distances = np.array(trimmed_distances, dtype=float)
        tap_count, taps_indices = self.count_taps(trimmed_distances)
        
        tap_count+=2
        # add first and last tap
        taps_indices = np.append(taps_indices, 0)
        taps_indices = np.append(taps_indices, len(trimmed_distances)-1)
        taps_indices = np.sort(taps_indices)
        
        max_amplitude_positions, amplitudes = self.calculate_amplitudes(trimmed_distances, taps_indices)

        # Calculate interval variation and tapping frequency
        interval_variation = np.std(np.diff(taps_indices) / self.fps)
        interval_mean = np.mean(np.diff(taps_indices) / self.fps)
        task_duration = (taps_indices[-1] - taps_indices[0]) / self.fps if len(taps_indices) > 1 else 0
        tapping_frequency = self.calculate_tapping_frequency(tap_count, task_duration)

        # Extract FFT features
        peak_frequency, peak_amplitude, power, num_peaks, freq_consistency  = self.calculate_fft_features(trimmed_distances)

        # Amplitude decrement and frequency decrement
        amplitude_decrement, frequency_decrement, mean_tap_speed, halts_and_hesitations = self.calculate_decrements(amplitudes, taps_indices, self.fps)

        # Calculate consistency
        consistency = np.var(trimmed_distances)

        # Calculate MSE for intervals
        interval_mse = np.sqrt(np.mean((np.diff(taps_indices) / self.fps - interval_mean) ** 2)) if len(taps_indices) > 1 else 0

        # Compile common features
        features = {
            'tap_count': tap_count,
            'interval_variation': interval_variation,
            'interval_mean': interval_mean,
            'task_duration': task_duration,
            'tapping_frequency': tapping_frequency,
            'tapping_start_time': start_index / self.fps,
            'interval_mse': interval_mse,
        }

        # Extract additional features for each type of distance: FINGER_NORMALIZED_DISTANCE, FINGER_DISTANCE, ANGULAR_DISTANCE
        for distance_type in [FINGER_NORMALIZED_DISTANCE, FINGER_DISTANCE, ANGULAR_DISTANCE]:
            distance_series = distances[distance_type][start_index:last_index + 1]
            max_amplitude_positions, amplitudes = self.calculate_amplitudes(distance_series, taps_indices)

            mean_amplitude = np.mean(amplitudes) if amplitudes else 0
            std_amplitude = np.std(amplitudes) if amplitudes else 0
            max_amplitude = np.max(amplitudes) if amplitudes else 0
            amplitude_mse = np.sqrt(np.mean((amplitudes - mean_amplitude) ** 2)) if amplitudes else 0

            amplitude_decrement, frequency_decrement, mean_tap_speed, halts_and_hesitations = self.calculate_decrements(amplitudes, taps_indices, self.fps)

            # FFT features
            peak_frequency, peak_amplitude, power, num_peaks, freq_consistency = self.calculate_fft_features(distance_series)

            # Catch22 features
            catch22_features = catch22_all(np.array(distance_series))

            # Add features to dictionary
            features.update({
                f'{distance_type}_mean_amplitude': mean_amplitude,
                f'{distance_type}_std_amplitude': std_amplitude,
                f'{distance_type}_amplitude_mse': amplitude_mse,
                f'{distance_type}_max_amplitude': max_amplitude,
                f'{distance_type}_amplitude_decrement': amplitude_decrement,
                f'{distance_type}_frequency_decrement': frequency_decrement,
                f'{distance_type}_mean_tap_speed': mean_tap_speed,
                f'{distance_type}_halts_and_hesitations': halts_and_hesitations,
                f'{distance_type}_consistency': consistency,
                f'{distance_type}_peak_frequency': peak_frequency,
                f'{distance_type}_peak_amplitude': peak_amplitude,
                f'{distance_type}_power': power,
                f'{distance_type}_num_peaks': num_peaks,
                f'{distance_type}_frequency_consistency': freq_consistency,
            })

            #Add catch22 features to the dictionary
            if distance_type == FINGER_NORMALIZED_DISTANCE:
                for key, value in zip(catch22_features['names'], catch22_features['values']):
                    features[f'{key}'] = value

        # Add wrist mobility features
        features.update({
            'wrist_total_distance': total_distance,
            'wrist_mean_speed': mean_speed,
            'wrist_std_speed': std_speed,
        })

        # Write features to a CSV file if output path is provided
        if output_path:
            features_df = pd.DataFrame([features])
            features_df.to_csv(output_path, index=False)

        return features


    def process_dataset(self, dataset_folder, output_file, recalculate=False):
        """
        Process the dataset folder and calculate features for each file.
        If output_file already exists, only calculate features for rows that are not in the output_file.
        """
        # Check if output file exists and load it
        if not recalculate and os.path.exists(output_file):
            print(f"Output file {output_file} exists. Loading existing data...")
            existing_features_df = pd.read_csv(output_file)
            processed_files = set(existing_features_df['file_name'].values)
        else:
            print(f"Output file {output_file} does not exist. Processing all files...")
            existing_features_df = pd.DataFrame()  # Empty dataframe to append new data
            processed_files = set()

        all_features = []

        # Loop over the dataset folder and process each file
        for root, _, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith('.csv') and file not in processed_files:
                    print(f'Processing {file}')
                    file_path = os.path.join(root, file)
                    try:
                        
                        distances = pd.read_csv(file_path)[['Frame', 'Finger Distance', 'Finger Normalized Distance', 'Angular Distance', 'Wrist Coordinate']]
                        if len(distances) < 5:
                            print(f"File {file} has less than 5 frames. Skipping...")
                            continue
                        features = self.extract_features_from_distances(distances)
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
                    #add file name
                    features['file_name'] = file
                    all_features.append(features)

        # Convert new features to DataFrame
        new_features_df = pd.DataFrame(all_features)


        # Append new features to existing features
        final_features_df = pd.concat([existing_features_df, new_features_df], ignore_index=True)

        # Save the final combined features to output file
        final_features_df.to_csv(output_file, index=False)

        return final_features_df

if __name__ == "__main__":
    
    base_processed_directory = Path(BASE_PROCESSED_DIRECTORY)
    feature_extractor = FeatureExtractor(base_processed_directory=base_processed_directory,fps=60)
    
    for task in ['left', 'right']:
        dataset_folder = base_processed_directory / 'finger_tapping' / task / 'distances'
        output_file = base_processed_directory / 'finger_tapping' / task / 'csvs' /  f'{task}_features.csv'
        feature_extractor.process_dataset(dataset_folder=dataset_folder, output_file=output_file, recalculate=True)
        print("Feature extraction completed.")
