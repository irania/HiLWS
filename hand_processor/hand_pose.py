import os
import sys
# Adjust the working directory to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)
os.chdir(project_root)
from concurrent.futures import ThreadPoolExecutor
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from config import BASE_PROCESSED_DIRECTORY

class HandPoseExtractor:
    def __init__(self, base_processed_directory):
        """
        Initialize the HandPoseExtractor with the base directory where videos are stored.
        """
        self.base_processed_directory = Path(base_processed_directory)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_hand_dimensions(self, landmarks):
        """
        Calculate the bounding box dimensions of the hand from landmarks.
        """
        try:
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            return hand_width, hand_height
        except Exception as e:
            print(f"Error calculating hand dimensions: {e}")
            return None, None

    def process_video(self, video_path, output_path):
        """
        Process a single video and extract hand pose data.
        """
        if output_path.exists():
            print(f"Skipping {video_path.name}, already processed.")
            return

        cap = cv2.VideoCapture(str(video_path))
        frame_results = []
        frame_number = 0
        # Initialize your timestamp tracker
        last_timestamp = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Get the current timestamp (you can also use frame number as a timestamp)
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get timestamp in milliseconds

            # Check if the current timestamp is greater than the last processed one
            if last_timestamp is None or current_timestamp > last_timestamp:
                last_timestamp = current_timestamp  # Update last_timestamp
            results = self.mp_hands.process(frame_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_id, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    hand_label = 'Left' if handedness.classification[0].label == 'Right' else 'Right'
                    hand_width, hand_height = self.calculate_hand_dimensions(hand_landmarks.landmark)

                    if hand_width is not None and hand_height is not None:
                        landmarks = [frame_number, hand_id, hand_label, hand_width, hand_height]
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        frame_results.append(landmarks)

                    if hand_id == 1:
                        print(f"2 hands detected in frame {frame_number}")

            frame_number += 1

        cap.release()

        if frame_results:
            # Define column names for landmarks
            columns = ['frame_number', 'hand_id', 'hand_label', 'hand_width', 'hand_height']
            for i in range(21):
                columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

            # Save the data to a CSV file
            df = pd.DataFrame(frame_results, columns=columns)
            df.to_csv(output_path, index=False)
            print(f"Processed {video_path.name}")
    
    def process_task(self, task, subtask):
        """
        Process all videos for a specific task (left or right hand) and save the extracted data.
        """
        video_directory = self.base_processed_directory / task / subtask / 'videos'
        output_directory = self.base_processed_directory / task / subtask / 'pose'

        # Create the output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)

        # Process each video in the directory
        for video_file in video_directory.iterdir():
            if video_file.suffix == '.mp4':
                output_path = output_directory / f"{video_file.stem}.csv"
                self.process_video(video_file, output_path)


    def close(self):
        """
        Close the MediaPipe hands object when done.
        """
        self.mp_hands.close()
        print("MediaPipe hands model closed.")


if __name__ == "__main__":
    base_processed_directory = BASE_PROCESSED_DIRECTORY
    hand_pose_extractor = HandPoseExtractor(base_processed_directory)
    #hand_pose_extractor.process_task("finger_tapping","left")
    #hand_pose_extractor.process_task("finger_tapping","right")
    hand_pose_extractor.process_task("hand_movement","left_open_close")
    hand_pose_extractor.close()