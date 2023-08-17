from flask import Flask, jsonify, request
import json
# from tensorflow.keras.preprocessing import image
import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import matplotlib.pyplot as plt
from PIL import Image
from base64 import b64decode, b64encode
from io import BytesIO
import cv2
from math import acos, pi, sqrt 
import mediapipe as mp

app = Flask(__name__)

def findDistance(x1, y1, x2, y2):
    dist = sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = acos((y2 -y1)*(-y1) / (sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/pi)*theta
    return degree


@app.route(
    '/predict',methods=['GET']
)

def predict():

    event = json.loads(request.data)
    values = event['image']
    decoded_image = b64decode(values)
    image_np = np.frombuffer(decoded_image, dtype=np.uint8)

    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Load your input image using OpenCV
    # input_image = cv2.imread(decoded_image)  # Replace 'input_image.jpg' with your image path
    input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # Get height and width.
    h, w = input_image.shape[:2]

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Process the image.
    keypoints = pose.process(image_rgb)

    # Convert the image back to BGR.
    input_image_with_keypoints = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    # Define keypoint labels.
    keypoint_labels = {
        lmPose.LEFT_SHOULDER: 'Left Shoulder',
        lmPose.RIGHT_SHOULDER: 'Right Shoulder',
        lmPose.LEFT_EAR: 'Left Ear',
        lmPose.LEFT_HIP: 'Left Hip'
    }


    # Acquire the landmark coordinates.
    # Once aligned properly, left or right should not be a concern.
    # Left shoulder.
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
    # Right shoulder
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
    # Left ear.
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
    # Left hip.
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Calculate angles (similar to your code).
    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Define angle thresholds for good and bad posture.
    good_neck_inclination_threshold = 40
    good_torso_inclination_threshold = 10

    # Determine whether it's a good or bad posture.
    if neck_inclination < good_neck_inclination_threshold and torso_inclination < good_torso_inclination_threshold:
        posture_text = 'Good Posture'
        posture_color = (0, 255, 0)  # Green
        draw_correction_lines = False
    else:
        posture_text = 'Bad Posture'
        posture_color = (0, 0, 255)  # Red
        draw_correction_lines = True

    # Draw the posture evaluation text on the image.
    cv2.putText(input_image_with_keypoints, posture_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, posture_color, 2)

   

    # Draw landmarks.
    cv2.circle(input_image_with_keypoints, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)  # Yellow
    cv2.circle(input_image_with_keypoints, (r_shldr_x, r_shldr_y), 7, (0, 192, 255), -1)  # Light Orange
    cv2.circle(input_image_with_keypoints, (l_ear_x, l_ear_y), 7, (0, 255, 255), -1)  # Yellow
    cv2.circle(input_image_with_keypoints, (l_hip_x, l_hip_y), 7, (0, 255, 255), -1)  # Yellow

    # Draw lines connecting keypoints.
    cv2.line(input_image_with_keypoints, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 0), 4)  # Green
    cv2.line(input_image_with_keypoints, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), (255, 0, 0), 4)  # Blue
    cv2.line(input_image_with_keypoints, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), (0, 0, 255), 4)  # Red

    
    image = input_image_with_keypoints
    


    #Decoding the image into base64 start ----------------------------------------------------------------------------
    encoded_image = b64encode(image).decode('utf8')
    # Decoding the image into base64 end ----------------------------------------------------------------------------

    return jsonify({"Posture": str(posture_text), "Image":encoded_image})
   


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000')




