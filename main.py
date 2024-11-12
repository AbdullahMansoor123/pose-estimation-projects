import os
from ultralytics import YOLO
import cv2
import numpy as np



# Angle Calculator
def calculate_angle(p1, p2, p3):
    """Calculate angle between 3 points"""
    # p1, p2, p3 are the points in format [x, y]
    # Calculate the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    if not np.isnan(angle_deg):
        angle_deg
        return int(angle_deg)
    else:
        pass 
    

def kpxy(image,point):
    '''get keypoint in x,y with respect to image'''
    x, y = point[0], point[1]
    h, w = image.shape[:2]
    px, py = int(x * w), int(y * h)
    return [px, py]


def visual_keypoints(image,keypoints):
    """ visualize keypoints on image"""
    for keypoint in keypoints:
        # Iterate over key points and add numbering
        for i, (x, y) in enumerate(keypoint):
            # Convert normalized coordinates (x, y) to pixel values based on image dimensions
            h, w = image.shape[:2]
            px, py = int(x * w), int(y * h)
            # Draw keypoint
            cv2.circle(image, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
            
            # Add text label with number
            cv2.putText(image, str(i + 1), (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return image



def add_text_top_left(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 255), thickness=2):
    """
    Adds text to the top-left corner of an image.
    """
    # Define the text position
    #position = (10, 30)  # Slight offset from the top-left corner

    # Add text to the image
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image



def check_posture(image, keypoint, box, postures_to_check=None):
    """
    postures_to_check = ["back", "shoulder", "leg"]

    body_keypoints = [(0, "Nose"), (1, "L_Eye"), (2, "R_Eye"), (3, "L_Ear"), (4, "R_Ear"), 
                    (5, "L_Shldr"), (6, "R_Shldr"), (7, "L_Elbow"), (8, "R_Elbow"), 
                    (9, "L_Wrist"), (10, "R_Wrist"), (11, "L_Hip"), (12, "R_Hip"), 
                    (13, "L_Knee"), (14, "R_Knee"), (15, "L_Ankle"), (16, "R_Ankle")]

    """
    if postures_to_check is None:
        print("Add body posture to check")
    
    def evaluate_posture_condition(image,p1, p2, p3, angle_name):
        angle = calculate_angle(p1, p2, p3)
        if angle not in range(90, 121):
            response_text = f"{angle}"
            image = add_text_top_left(image, text=response_text, position=p2, color=(0, 0, 255), font_scale=0.7)
        elif np.isnan(angle):
            image
        else:
            response_text = f"{angle}"
            image = add_text_top_left(image, text=response_text, position=p2, color=(0, 255, 0), font_scale=0.7)
        return image

    # box_tl = (int(box[0]), int(box[1])+20)

    if "back" in postures_to_check:
        # Back posture condition
        image = evaluate_posture_condition(image,
            kpxy(image, keypoint[6]),
            kpxy(image, keypoint[12]),
            kpxy(image, keypoint[14]),
            "BA"
        )

    if "shoulder" in postures_to_check:
        # Shoulder posture condition
        image = evaluate_posture_condition(image,
            kpxy(image, keypoint[6]),
            kpxy(image, keypoint[8]),
            kpxy(image, keypoint[10]),
            "AA"
        )

    if "leg" in postures_to_check:
        # Leg posture condition
        image = evaluate_posture_condition(image,
            kpxy(image, keypoint[12]),
            kpxy(image, keypoint[14]),
            kpxy(image, keypoint[16]),
            "leg"
        )
    return image


# Load pose model
model = YOLO("yolo11n-pose.pt")  # load an official pose model

# Open video file
cap = cv2.VideoCapture("test_videos\\my_video.mp4")

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec if needed
out = cv2.VideoWriter('my_output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Loop over frames

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    # Predict with the model
    results = model(image)[0]  # Predict on an image
    boxes = results.boxes.xyxy.tolist()

    # Keypoints object for pose outputs
    keypoints = results.keypoints.xyn.tolist()
    image = results.plot(boxes=False, kpt_radius=2)
    
    # UNCOMMIT to visualize keypoint numbering
    # image = visual_keypoints(image,keypoints)

    # Process keypoints to check posture for different body parts
    for keypoint, box in zip(keypoints,boxes):
        image = check_posture(image, keypoint, box, postures_to_check=['back','shoulder','leg'])

    # Display the frame with pose overlay
    # cv2.imshow("Pose Estimation", cv2.resize(image, (720, 720)))
    # # Write the frame with pose drawing to the output video
    out.write(image)
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()