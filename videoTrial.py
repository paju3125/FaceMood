# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load pre-trained model
# model = load_model('./models/base_1_overfit.h5')  # Replace with your actual model file

# print(model.summary())
# # Open video file
# video_capture = cv2.VideoCapture('anger1.mp4')  # Replace with your video file

# while True:
#     # Read frame
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Preprocess frame
#     frame = cv2.resize(frame, (48, 48))  # Adjust size according to your model's input size
#     frame = np.expand_dims(frame, axis=0)  # Add batch dimension
#     frame = frame / 255.0  # Normalize pixel values

#     # Predict emotion
#     emotion_probabilities = model.predict(frame)
#     print(emotion_probabilities)
#     predicted_emotion = np.argmax(emotion_probabilities)

#     # Display or save results as needed
#     print("Predicted Emotion:", predicted_emotion)

# # Release video capture object
# video_capture.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained emotion model
emotion_model = load_model('./models/base_1_overfit.h5')  # Replace with the actual path to your model

# Function to normalize coordinates to pixel coordinates
def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    x = int(normalized_x * image_width)
    y = int(normalized_y * image_height)
    return x, y

# Load mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

# Function to resize image if needed
def rescale():
    # Add your logic for rescaling here
    return True  # Adjust this based on your requirements

# Mapping of emotion index to emotion label
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Open video file
video_path = 'VEGEMITE62_2.mp4'  # Replace with the actual path to your video file
video = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = video.get(5)

# Create an output video file
output_path = 'output_video.mp4'  # Replace with the desired output path
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

predicted_emotions_list = []

# Process each frame in the video
while True:
    ret, frame = video.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face detection
    results = face_detection.process(img)

    image_rows, image_cols, _ = frame.shape

    if results.detections:
        for detection in results.detections:
            try:
                box = detection.location_data.relative_bounding_box

                x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols,
                                                    image_rows)

                # Crop image to face
                cimg = frame[x[1]:y[1], x[0]:y[0]]
                if rescale():
                    cropped_img = np.expand_dims(cv2.resize(cimg, (48, 48)), 0)
                else:
                    cropped_img = np.expand_dims(cv2.resize(cimg, (48, 48)), 0) / 255.

                # Get model prediction
                pred = emotion_model.predict(cropped_img)
                idx = int(np.argmax(pred))
                confidence = np.max(pred)

                # Store predicted emotion and confidence in the list
                predicted_emotions_list.append({'emotion': emotion_dict[idx], 'confidence': confidence})

                # Display emotion text and confidence on the frame
                text = f"{emotion_dict[idx]} ({confidence:.2f})"
                cv2.putText(frame, text, (x[0], y[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error processing frame: {e}")
                pass

    # Write the processed frame to the output video file
    output_video.write(frame)

# Release video capture and output video writer objects
video.release()
output_video.release()
cv2.destroyAllWindows()

# Analyze the list to find the most predicted emotion with the highest confidence
most_common_emotion = max(predicted_emotions_list, key=lambda x: x['confidence'])
print(f"The most predicted emotion is: {most_common_emotion['emotion']} with confidence: {most_common_emotion['confidence']:.2f}")