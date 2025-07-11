from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, find_peaks
FINGER_NORMALIZED_DISTANCE = 'Finger Normalized Distance'
WRIST_COORDINATE = 'Wrist Coordinate'
class TapTrimmer:

    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir

    def count_taps(self, distances, min_prominence=0.15, distance=10, height_multiplier=0.3):
        distances = np.array(distances, dtype=float)
        taps_indices = argrelextrema(distances, np.less)[0]
        peaks, _ = find_peaks(-distances, prominence=min_prominence, distance=distance, height=-height_multiplier * max(distances))
        taps_indices = np.intersect1d(taps_indices, peaks)
        tap_count = len(taps_indices)
        return tap_count, taps_indices

    def find_best_segment_indices(self, distances, taps_indices):
        # Implement your logic to find the best segment 
        if (taps_indices.size == 0):
            return 0, len(distances) - 1
        start_index = taps_indices[0]
        last_index = taps_indices[-1]
        return start_index, last_index

    def extract_features_from_distances(self, distances, output_path=None):
        """
        Extract all features (tapping, amplitudes, FFT, wrist mobility) from the distance data.
        """
        # Extract best segment indices for trimming
        tap_count, taps_indices = self.count_taps(distances[FINGER_NORMALIZED_DISTANCE])
        if tap_count == 0:
            return
        start_index, last_index = self.find_best_segment_indices(distances[FINGER_NORMALIZED_DISTANCE], taps_indices)

        # Trim distances and pose data
        trimmed_distances = distances[FINGER_NORMALIZED_DISTANCE][start_index:last_index + 1]


        # Extract tapping and amplitude features
        trimmed_distances = np.array(trimmed_distances, dtype=float)
        tap_count, taps_indices = self.count_taps(trimmed_distances)

        # Segment and save each tap from the trimmed video
        self.segment_and_save_taps(start_index, last_index, taps_indices)

    def segment_and_save_taps(self, start_index, last_index, taps_indices):
        cap = cv2.VideoCapture(self.video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

        # Iterate over tap indices in pairs to define start and end of each segment
        for i in range(len(taps_indices) - 1):
            start_frame = taps_indices[i]
            end_frame = taps_indices[i + 1]

            if start_frame < start_index or end_frame > last_index:
                # Skip indices outside of the trimmed video segment
                continue

            # Set the video capture to the start frame of the segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            segment_path = f"{self.output_dir}/tap_segment_{i + 1}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(segment_path, fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            print(f"Saving segment {i + 1} from frame {start_frame} to {end_frame}")

            # Read and write frames between the start and end frames
            for frame_no in range(start_frame, end_frame):
                success, frame = cap.read()
                if success:
                    out.write(frame)
                else:
                    break

            out.release()

        cap.release()



# Example usage
base_processed_directory = Path(r'\\files.ubc.ca\team\PPRC\Camera\Booth_Processed')

for hand in ['left', 'right']:
    video_dir = base_processed_directory / 'finger_tapping' / f'{hand}' / 'videos' 
    output_dir = base_processed_directory / 'finger_tapping' / f'{hand}' / 'trimmed_taps' 
    distance_dir = base_processed_directory / 'finger_tapping' / f'{hand}'/ 'distances' 
    
    for video_file in video_dir.iterdir():
        if video_file.suffix == '.mp4':
            distances = pd.read_csv(distance_dir / f"{video_file.stem}_distances.csv")
            trim_out_dir = output_dir / f'{video_file.stem}'
            if  trim_out_dir.exists():
                continue
            trim_out_dir.mkdir(parents=True, exist_ok=True)
            processor = TapTrimmer(video_file, trim_out_dir)
            processor.extract_features_from_distances(distances)
