import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between two vectors
def calculate_angle_between_vectors(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_u, v2_u)
    angle = np.arccos(dot_product) * (180.0 / np.pi)  # Convert to degrees
    return angle

# Start capturing video
cap = cv2.VideoCapture(r"C:\Users\atefe\Pictures\Camera Roll\WIN_20241029_10_15_02_Pro.mp4")  # Replace with 0 for webcam or file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract relevant landmarks (wrist, index finger tip, and pinky tip)
            wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                              hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                              hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z])

            index_knuckle = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z])

            pinky_knuckle = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z])

            # Convert normalized coordinates to pixel coordinates
            wrist_px = (int(wrist[0] * width), int(wrist[1] * height))
            index_px = (int(index_knuckle[0] * width), int(index_knuckle[1] * height))
            pinky_px = (int(pinky_knuckle[0] * width), int(pinky_knuckle[1] * height))

            # Calculate plane normal (hand orientation) using cross product
            vec_wrist_index = index_knuckle - wrist
            vec_wrist_pinky = pinky_knuckle - wrist
            hand_normal = np.cross(vec_wrist_index, vec_wrist_pinky)

            # Define camera axis (z-axis facing the camera)
            camera_axis = np.array([0, 1, 0])
            angle = calculate_angle_between_vectors(hand_normal, camera_axis)

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Color the plane by drawing a filled triangle between the three points
            pts = np.array([wrist_px, index_px, pinky_px], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.fillPoly(frame, [pts], color=(0, 255, 255, 50))

            # Display the z-axis vector (from wrist point upwards in the image)
            # z_axis_end = (wrist_px[0], wrist_px[1] - 100)
            # cv2.arrowedLine(frame, wrist_px, z_axis_end, (255, 0, 0), 2, tipLength=0.3)
            # cv2.putText(frame, 'Z-axis', (z_axis_end[0] + 10, z_axis_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display the calculated angle on the frame
            cv2.putText(frame, f'Angle: {angle:.2f} degrees', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

    # Show the video frame
    cv2.imshow('Hand Pose Angle and Plane Visualization', frame)
    cv2.waitKey(10)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
