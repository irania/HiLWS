import os
from pathlib import Path
import pandas as pd
import numpy as np
import math
import csv
import mediapipe as mp
import math


class DistanceCalculator:
    def __init__(self, width=1920, height=1080, base_processed_directory=None):
        """
        Initialize DistanceCalculator with width and height for normalization.
        """
        self.width = width
        self.height = height
        self.base_processed_directory = Path(base_processed_directory) if base_processed_directory else None
        self.mp_hands = mp.solutions.hands
        self.distance_names = ['Frame', 'Finger Distance', 'Finger Normalized Distance', 'Angular Distance', 'Wrist Coordinate', 'Hand BBox Width', 'Hand BBox Height']

    def _scale_landmark(self, landmark):
        """
        Scale landmark coordinates according to width, height, and keep z as-is (or scale if needed).
        """
        x_scaled = landmark[0] * self.width
        y_scaled = landmark[1] * self.height
        z_scaled = landmark[2] * self.width  # scale z to match x units
        return np.array([x_scaled, y_scaled, z_scaled])

    def calculate_finger_distance(self, hand_landmarks):
        """
        Calculate the 3D distance between thumb tip and index finger tip.
        """
        thumb_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP.value])
        index_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance

    def calculate_normalized_finger_distance(self, hand_landmarks):
        """
        Calculate normalized 3D distance between thumb tip and index finger tip.
        """
        thumb_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP.value])
        index_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
        wrist = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.WRIST.value])
        thumb_cmc = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.THUMB_CMC.value])

        finger_distance = np.linalg.norm(thumb_tip - index_tip)
        wrist_to_cmc_distance = np.linalg.norm(wrist - thumb_cmc)

        if wrist_to_cmc_distance == 0:
            wrist_to_cmc_distance = 1  # Avoid division by zero

        return finger_distance / wrist_to_cmc_distance

    def calculate_angular_distance(self, hand_landmarks):
        """
        Calculate the angle at the wrist formed by thumb tip and index finger tip in 3D.
        """
        wrist = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.WRIST.value])
        thumb_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.THUMB_TIP.value])
        index_tip = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP.value])

        vec_wt = thumb_tip - wrist
        vec_wi = index_tip - wrist

        dot = np.dot(vec_wt, vec_wi)
        norm_product = np.linalg.norm(vec_wt) * np.linalg.norm(vec_wi)

        if norm_product == 0:
            return 0.0

        cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))

        return angle

    def calculate_wrist_coordinates(self, hand_landmarks):
        """
        Return wrist coordinates (x, y, z) scaled.
        """
        wrist = self._scale_landmark(hand_landmarks[self.mp_hands.HandLandmark.WRIST.value])
        return tuple(wrist)

    def calculate_distances(self, pose_output_path):
        """
        Process CSV files to calculate distances and save the results.
        """
        pose_output_path = Path(pose_output_path)
        distance_output_folder = pose_output_path.parent.parent / 'distances'
        distance_output_folder.mkdir(parents=True, exist_ok=True)

        output_path = distance_output_folder / f"{pose_output_path.stem}_distances.csv"

        df = pd.read_csv(pose_output_path)

        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.distance_names)

            for frame_number in df['frame_number'].unique():
                frame_data = df[df['frame_number'] == frame_number]
                if len(frame_data) == 0:
                    continue

                hand_landmarks = []
                for i in range(21):
                    x = frame_data[f'x_{i}'].values[0]
                    y = frame_data[f'y_{i}'].values[0]
                    z = frame_data[f'z_{i}'].values[0]
                    hand_landmarks.append([x, y, z])

                if all([x == 0 and y == 0 and z == 0 for x, y, z in hand_landmarks]):
                    continue

                distance = self.calculate_finger_distance(hand_landmarks)
                normalized_distance = self.calculate_normalized_finger_distance(hand_landmarks)
                angular_distance = self.calculate_angular_distance(hand_landmarks)
                wrist_coordinates = self.calculate_wrist_coordinates(hand_landmarks)
                bbox_width = frame_data['hand_width'].values[0] * self.width
                bbox_height = frame_data['hand_height'].values[0] * self.height

                writer.writerow([frame_number, distance, normalized_distance, angular_distance, wrist_coordinates, bbox_width, bbox_height])

        print(f"Distances processed and saved to {output_path}")
        return output_path


    def read_distances(self, distance_file):
        """
        Read the distances from the CSV file.
        """
        distances = pd.read_csv(distance_file)[self.distance_names]
        return distances
    
def process_booth_folders(distance_calculator):
    """
    Process CSV files in the booth folders for both 'right' and 'left' tasks and calculate distances.
    """
    if not distance_calculator.base_processed_directory:
        raise ValueError("Base processed directory is not set.")

    for task in ['right', 'left']:
        # Define folder paths
        csv_folder = distance_calculator.base_processed_directory / 'finger_tapping' / task / 'pose'
        output_folder = distance_calculator.base_processed_directory / 'finger_tapping' / task / 'distances'
        
        # Ensure output folder exists
        output_folder.mkdir(parents=True, exist_ok=True)

        # Process each CSV file in the folder
        for csv_file in csv_folder.iterdir():
            if csv_file.suffix == '.csv':
                output_path = output_folder / f"{csv_file.stem}_distances.csv"

                # Check if the output CSV already exists
                if not output_path.exists():
                    print(f"Processing {csv_file.name} for task {task}")

                    # Use the DistanceCalculator to calculate distances and save to output CSV
                    distance_calculator.calculate_distances(csv_file)

                    print(f"Processed {csv_file.name}, saved to {output_path}")
                        
                        
                        
                        
if __name__ == "__main__":
    #base_processed_directory = Path("processed_data")
    base_processed_directory = r'D:\Datasets\Parkinson\Finger Tapping Data'
    distance_calculator = DistanceCalculator(base_processed_directory=base_processed_directory, width=640, height=480)

    # Process booth folders
    process_booth_folders(distance_calculator)

