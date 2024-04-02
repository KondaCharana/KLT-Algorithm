

import cv2
import numpy as np

# Read input video
cap = cv2.VideoCapture('D:\SORT\rudra.mp4')


# Check if the video capture is open
if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()

# Read the first frame
ret, old_frame = cap.read()

# Check if the frame is empty
if not ret:
    print("Error: Failed to read first frame from the video.")
    exit()

# Convert the first frame to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Define the KLT parameters
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detect keypoints in the first frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Calculate the bounding box around all keypoints
    if len(good_new) > 0:
        min_x = int(np.min(good_new[:, 0]))
        max_x = int(np.max(good_new[:, 0]))
        min_y = int(np.min(good_new[:, 1]))
        max_y = int(np.max(good_new[:, 1]))

        # Draw bounding box on the frame
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    output_frame = frame
    out.write(output_frame)  # Write the frame to the output video

    cv2.imshow('frame', output_frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Update the previous frame and previous keypoints
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

