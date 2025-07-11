import pandas as pd
import os

BASE_PROCESSED_DIRECTORY = r'\\files.ubc.ca\team\PPRC\Camera\Booth_Processed'
output_directory = rf'{BASE_PROCESSED_DIRECTORY}\finger_tapping\right\pose'

def analyze_video_quality(file_path):
    try:
        df = pd.read_csv(file_path)
        
        filename = os.path.basename(file_path)
        
        # Two hands detection and check if labels are different
        two_hands_count = 0
        two_hands_up_detected = 0
        for frame_number in df['frame_number'].unique():
            frame_data = df[df['frame_number'] == frame_number]
            if len(frame_data) > 1:
                labels = frame_data['hand_label'].unique()
                if len(labels) > 1:
                    two_hands_up_detected += 1
                two_hands_count += 1
        
        # Frames where hands are not detected are frames that theirr index is not in range max frame number
        #find difference between two arrqaays

        hand_not_detected_frames = [x for x in range(df['frame_number'].max()) if x not in df['frame_number'].unique()]
        duplicated_frames = df[df.duplicated(subset=['frame_number'])]
        
        # All frame numbers, remove duplicates and compare with max frame number
        all_frame_numbers = df['frame_number'].unique().tolist()
        max_frame_number = df['frame_number'].max()
        removed_frames_count = max_frame_number - len(all_frame_numbers) + 1 
        
        return {
            "filename": filename,
            "two_hands_count": two_hands_count,
            "two_hands_up": two_hands_up_detected > 0,
            'two_hands_frames': duplicated_frames['frame_number'].tolist(),
            "hand_not_detected_frames": hand_not_detected_frames,
            "hand_not_detected_count": removed_frames_count,
            
        }
    except Exception as e:
        print(f"Error analyzing video quality for {file_path}: {e}")
        return None

def generate_quality_report(quality_data,output_csv):
    report_df = pd.DataFrame(quality_data)
    report_path = os.path.join(output_csv, "video_quality.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Quality report saved to {report_path}")

if __name__ == '__main__':
    try:
        output_csv = rf'{BASE_PROCESSED_DIRECTORY}\finger_tapping\left\csvs'
        quality_data = []
        for filename in os.listdir(output_directory): 
            if filename.endswith('.csv'):
                file_path = os.path.join(output_directory, filename)
                result = analyze_video_quality(file_path)
                if result:
                    quality_data.append(result)

        generate_quality_report(quality_data, output_csv)

        print("Video quality analysis and report generation complete.")
    except Exception as e:
        print(f"Error in main execution: {e}")
