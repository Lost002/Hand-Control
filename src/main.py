import cv2
import mediapipe as mp

# Initialize mediapipe hands module
mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Use 0 for the default camera, or another number if you have multiple cameras
camera_index = 0

# Initialize video capture with the camera index
vidcap = cv2.VideoCapture(camera_index)

# Set the desired window width and height
winwidth = 1280
winheight = 720

# Initialize hand tracking
with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        processFrames = hands.process(rgb_frame)

        # Draw landmarks on the frame
        if processFrames.multi_hand_landmarks:
            for lm in processFrames.multi_hand_landmarks:
                mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)
                
                #index_tip = lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                #middle_tip = lm.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                #print(f"index Tip Position: x={int(index_tip.x*100)}%, y={int(index_tip.y*100)}%")
                #print(f"middle Tip Position: x={int(middle_tip.x*100)}%, y={int(middle_tip.y*100)}%")
                #print(f"Thumb Tip Position: x={int(thumb_tip.x*100)}%, y={int(thumb_tip.y*100)}%")

        # Resize the frame to the desired window size
        resized_frame = cv2.resize(frame, (winwidth, winheight))

        # Display the resized frame
        cv2.imshow('Hand Tracking', resized_frame)

        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
vidcap.release()
cv2.destroyAllWindows()
