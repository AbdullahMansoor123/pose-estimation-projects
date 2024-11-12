import os
from ultralytics import YOLO
import cv2
import numpy as np

<<<<<<< HEAD
#keypoint map
# body_parts = {
#     0: "Nose",
#     1: "L_Eye",
#     2: "R_Eye",
#     3: "L_Ear",
#     4: "R_Ear",
#     5: "L_Shldr",
#     6: "R_Shldr",
#     7: "L_Elbow",
#     8: "R_Elbow",
#     9: "L_Wrist",
#     10: "R_Wrist",
#     11: "L_Hip",
#     12: "R_Hip",
#     13: "L_Knee",
#     14: "R_Knee",
#     15: "L_Ankle",
#     16: "R_Ankle"
# }


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
    
    return int(angle_deg)

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
    """
    if postures_to_check is None:
        print("Add body posture to check")
    
    def evaluate_posture_condition(p1, p2, p3, angle_name):
        angle = calculate_angle(p1, p2, p3)
        if angle not in range(90, 120):
            response_text = f"{angle_name}_{angle}"
            image = add_text_top_left(image, text=response_text, position=box_tl, color=(0, 0, 255))
        return image

    box_tl = (int(box[0]), int(box[1])+20)

    if "back" in postures_to_check:
        # Back posture condition
        image = evaluate_posture_condition(
            kpxy(image, keypoint[6]),
            kpxy(image, keypoint[12]),
            kpxy(image, keypoint[14]),
            "BA"
        )

    if "shoulder" in postures_to_check:
        # Shoulder posture condition
        image = evaluate_posture_condition(
            kpxy(image, keypoint[6]),
            kpxy(image, keypoint[8]),
            kpxy(image, keypoint[10]),
            "AA"
        )


    if "leg" in postures_to_check:
        # Leg posture condition
        image = evaluate_posture_condition(
            kpxy(image, keypoint[12]),
            kpxy(image, keypoint[14]),
            kpxy(image, keypoint[16]),
            "leg"
        )

    return image



=======
# Detection model (for person detection)
det_model = YOLO("yolov8n.pt")

>>>>>>> 47fc2bf2742ffa28dc0aedd844a370d8c8d548b9
# Load pose model
model = YOLO("yolo11n-pose.pt")  # load an official pose model

# Open video file
<<<<<<< HEAD
cap = cv2.VideoCapture("my_video.mp4")
=======
cap = cv2.VideoCapture("1.mp4")
>>>>>>> 47fc2bf2742ffa28dc0aedd844a370d8c8d548b9

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec if needed
<<<<<<< HEAD
out = cv2.VideoWriter('my_output_video_.avi', fourcc, 20.0, (frame_width, frame_height))
=======
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))
>>>>>>> 47fc2bf2742ffa28dc0aedd844a370d8c8d548b9

# Loop over frames

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    # Predict with the model
    results = model(image)[0]  # Predict on an image
    boxes = results.boxes.xyxy.tolist()

<<<<<<< HEAD
    # Keypoints object for pose outputs
    keypoints = results.keypoints.xyn.tolist()
    kp_image = results.plot(boxes=False, kpt_radius=2)
    image = visual_keypoints(kp_image,keypoints)

    # Process results list
    i=0
    for keypoint, box in zip(keypoints,boxes):
        image = check_posture(image, keypoint, box, postures_to_check=['back'])

    # Display the frame with pose overlay
    cv2.imshow("Pose Estimation", cv2.resize(image, (1080, 720)))
    # # Write the frame with pose drawing to the output video
    out.write(image)
=======
    # Run pose detection on the frame
    results = model(frame)
    keypoints = results[0].keypoints
    frame = results[0].plot(boxes=False, kpt_radius=2)

    # Write the frame with pose drawing to the output video
    out.write(frame)

    # Display the frame with pose overlay
    cv2.imshow("Pose Estimation", frame)

>>>>>>> 47fc2bf2742ffa28dc0aedd844a370d8c8d548b9
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> 47fc2bf2742ffa28dc0aedd844a370d8c8d548b9
