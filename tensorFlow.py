
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Define the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Resize the frame to 224x224
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score on the frame
    cv2.putText(frame, 
                f"Class: {class_name[2:].strip()}, Confidence: {confidence_score:.2f}", 
                (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5,  # Larger font size
                (0, 255, 0),  # Neon green color
                3,  # Thicker text
                cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)
    
    # Exit on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


