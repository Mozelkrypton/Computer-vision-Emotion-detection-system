import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

#This function loads the CNN model for emotion classification
def loading_model(model_path):
    return load_model(model_path)

#This function loads the Haar Cascade for face detection
def cascade_clasify(cascade_path):
    return cv2.CascadeClassifier(cascade_path)

#This function detects and classify emotions in the given image
def detect_classify_emotion(image_path, classifier, face_classifier, emotion_labels):
    # Reading the image from the given path that you have specified
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Here Convert the image to grayscale
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Detect faces in the image that has been converted

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Extract the face region
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # The line draws a rectangle around the face
        face = cv2.resize(face, (48, 48))  # Resize the face to 48x48 pixels so as to match the pre trained models size
        face = face.astype('float') / 255.0  
        face = img_to_array(face)  # Convert the face to an array
        face = np.expand_dims(face, axis=0)  # Add a batch dimension

        # Print the shape of the face array
        print(f"Face shape: {face.shape}")

        prediction = classifier.predict(face)[0]  # Predict the emotion

        # Debug: Print the prediction array
        print(f"Prediction: {prediction}")

        prediction_label = emotion_labels[prediction.argmax()]  # Get the label of the predicted emotion from the pre trained model

        # Debug: Print the predicted label
        print(f"Predicted label: {prediction_label}")

        cv2.putText(image, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)  # Displaying the emotion label

    # Display the image with the detected faces and emotion labels
    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# importation of various files from desktop
model_path = r'C:\Users\Admin\OneDrive\Desktop\pythonemotiond\model.h5'
cascade_path = r'C:\Users\Admin\OneDrive\Desktop\pythonemotiond\haarcascade_frontalface_default.xml'
image_path = r'C:\Users\Admin\OneDrive\Desktop\pythonemotiond\phhhp.jpg'

# Load the model and cascade classifier
classifier = loading_model(model_path)
face_classifier = cascade_clasify(cascade_path)

# Spesifyies the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Detect and classify emotion in the image
detect_classify_emotion(image_path, classifier, face_classifier, emotion_labels)
