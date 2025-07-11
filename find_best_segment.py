import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, argrelextrema
from scipy.fft import fft, fftfreq
from pycatch22 import catch22_all
from matplotlib import pyplot as plt
from config import BASE_PROCESSED_DIRECTORY

FINGER_DISTANCE = 'Finger Distance'
FINGER_NORMALIZED_DISTANCE = 'Finger Normalized Distance'
ANGULAR_DISTANCE = 'Angular Distance'
WRIST_COORDINATE = 'Wrist Coordinate'
FRAME = 'Frame'


class FingerTappingFeatureExtractor:
    def __init__(self, base_processed_directory=None, fps=60):
        self.base_processed_directory = base_processed_directory
        self.fps = fps

    def count_taps(self, distances, min_prominence=0.15, distance = 10, height_multiplyer = 0.3):
        distances = np.array(distances, dtype=float)
        taps_indices = argrelextrema(distances, np.less)[0]
        peaks, _ = find_peaks(-distances, prominence=min_prominence, distance=distance, height=-height_multiplyer*(max(distances)))
        taps_indices = np.intersect1d(taps_indices, peaks)
        tap_count = len(taps_indices)
        return tap_count, taps_indices

    def find_peaks_finger_tapping(self, data):
        """Finds tap indices (minima) in FINGER_NORMALIZED_DISTANCE."""
        distances = data[FINGER_NORMALIZED_DISTANCE].values
        tap_count, taps_indices = self.count_taps(distances)
        
        if len(taps_indices) == 0:
            return distances, [], [], 0, len(distances) - 1

        # Find best segment
        start_index, end_index = self.find_best_segment_indices(distances, taps_indices)

        # Trim tap indices to the segment and shift indices
        taps_indices = taps_indices[(taps_indices >= start_index) & (taps_indices <= end_index)]
        taps_indices -= start_index  # Shift indices to start from 0

        return distances, [], taps_indices, start_index, end_index

    def extract_features_from_distances(self, distances, output_path=None):
        """Extracts tapping features and writes them to a CSV file."""
        if len(distances) < 5:
            return None
        
        _, _, taps_indices, start_index, end_index = self.find_peaks_finger_tapping(distances)
        
        tap_indices_str = ",".join(map(str, taps_indices))  # Convert for CSV storage
        task_duration = (taps_indices[-1] - taps_indices[0]) / self.fps if len(taps_indices) > 1 else 0
        tapping_frequency = len(taps_indices) / task_duration if task_duration > 0 else 0

        features = {
            'tap_count': len(taps_indices),
            'start_index': start_index,
            'end_index': end_index,
            'task_duration': task_duration,
            'tapping_frequency': tapping_frequency,
            'tap_indices': tap_indices_str,
        }

        if output_path:
            pd.DataFrame([features]).to_csv(output_path, index=False)

        return features

    def process_dataset(self, dataset_folder, output_file, recalculate=False):
        """Processes dataset folder, extracting tapping features."""
        if not recalculate and os.path.exists(output_file):
            existing_features_df = pd.read_csv(output_file)
            processed_files = set(existing_features_df['file_name'].values)
        else:
            existing_features_df = pd.DataFrame()
            processed_files = set()

        all_features = []
        for root, _, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith('.csv') and file not in processed_files:
                    print(f'Processing {file}')
                    file_path = os.path.join(root, file)
                    try:
                        distances = pd.read_csv(file_path)[[FRAME, FINGER_DISTANCE, FINGER_NORMALIZED_DISTANCE, ANGULAR_DISTANCE, WRIST_COORDINATE]]
                        features = self.extract_features_from_distances(distances)
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        continue
                    
                    features['file_name'] = file
                    all_features.append(features)

        final_features_df = pd.concat([existing_features_df, pd.DataFrame(all_features)], ignore_index=True)
        final_features_df.to_csv(output_file, index=False)
        return final_features_df
    
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

class BestSegmentExtractor:
    def __init__(self, base_processed_directory, fps=60):
        self.feature_extractor = FingerTappingFeatureExtractor(base_processed_directory=base_processed_directory, fps=fps)

    def extract_and_plot_best_segments(self, dataset_folder, output_file, plot_dir):
        """Extracts and plots the best tapping segments."""
        os.makedirs(plot_dir, exist_ok=True)
        results = []

        for root, _, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        data = pd.read_csv(file_path)
                        if 'Frame' not in data or 'Finger Normalized Distance' not in data:
                            print(f"Skipping {file}: missing required columns.")
                            continue

                        distances, _, min_indices, start_index, end_index = self.feature_extractor.find_peaks_finger_tapping(data)

                        # Adjust tap indices based on segment start
                        tap_indices_str = ",".join(map(str, min_indices))

                        results.append({
                            'file_name': file,
                            'start_index': start_index,
                            'end_index': end_index,
                            'tap_indices': tap_indices_str
                        })

                        # self.plot_time_series_full(
                        #     file_name=file.replace('.csv', '.png'),
                        #     y=distances,
                        #     start_index=start_index,
                        #     end_index=end_index,
                        #     mins=min_indices,
                        #     title=f"Best Segment: {file}",
                        #     xlabel='Frame',
                        #     ylabel='Finger Normalized Distance',
                        #     output_dir=plot_dir
                        # )
                    except FileNotFoundError as e:
                        print(f"Error processing {file}: {e}")
                        continue

        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}. Plots stored in {plot_dir}")

    def plot_time_series_full(self, file_name, y, start_index, end_index, mins=None, output_dir=None, xlabel='Frame', ylabel='Distance'):
        """Plots time-series data with detected taps."""
        plt.figure(figsize=(10, 6))
        x = range(len(y))
        plt.plot(x, y, label=ylabel, color='blue')
        plt.plot(x[start_index:end_index + 1], y[start_index:end_index + 1], color='orange', label='Tapping Segment')

        if mins is not None:
            plt.scatter(mins, [y[i] for i in mins], color='red', label='Tap', marker='x')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(file_name)
        plt.legend()
        plt.grid(True)

        if output_dir:
            plt.savefig(os.path.join(output_dir, file_name))
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    base_processed_directory = Path(BASE_PROCESSED_DIRECTORY)
    best_segment_extractor = BestSegmentExtractor(base_processed_directory=base_processed_directory, fps=60)

    for task in ['left', 'right']:
        dataset_folder = base_processed_directory / 'finger_tapping' / task / 'distances'
        output_file = base_processed_directory / 'finger_tapping' / task / 'csvs' / f'{task}_best_segments_and_indexes.csv'
        plot_dir = base_processed_directory / 'finger_tapping' / task / 'plots'
        best_segment_extractor.extract_and_plot_best_segments(dataset_folder=dataset_folder, output_file=output_file, plot_dir=plot_dir)
