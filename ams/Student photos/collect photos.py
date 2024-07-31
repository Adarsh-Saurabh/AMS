import cv2
import os

def capture_and_save_photos(person_name, num_photos):
    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a folder with the person's name if it doesn't exist
    os.makedirs(person_name, exist_ok=True)

    # Capture photos from webcam and save them
    cap = cv2.VideoCapture(0)
    count = 0
    photo_count = 0

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Crop the face from the original image
                cropped_face = frame[y:y+h, x:x+w]

                # Save the cropped face
                photo_count += 1
                output_path = os.path.join(person_name, f"{person_name}_{photo_count}.jpg")
                cv2.imwrite(output_path, cropped_face)
                print(f"Saved: {output_path}")

                # Check if the required number of photos are captured
                if photo_count == num_photos:
                    break

        # Break the loop if the required number of photos are captured
        if photo_count == num_photos:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Ask the user to enter the name of the folder
folder_name = input("Enter the name of the folder to save the photos: ")

# Capture and save photos
capture_and_save_photos(folder_name, num_photos=150)
