import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def capture_and_save_photos(person_name, num_photos):
    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a folder with the person's name if it doesn't exist
    os.makedirs( f"./people/{person_name}", exist_ok=True)

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
                output_path = os.path.join(f"./people/{person_name}", f"{person_name}_{photo_count}.jpg")
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






import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from tensorflow.keras import layers, models, regularizers


def add_gaussian_blur(image):
    # Convert to an OpenCV image (uint8 format)
    image = np.array(image, dtype=np.uint8)
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)  # (5, 5) is the kernel size, 0 is the sigma
    return image

def ml_function():
    main_directory = "./people"

    # Parameters
    img_height, img_width = 100, 100  # Input size for ResNetV2
    batch_size = 32
    epochs = 20
    num_classes = len(os.listdir(main_directory))  # Number of class folders

    # Simple ImageDataGenerator without augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )

    # Generate batches of data from the directories
    train_generator = datagen.flow_from_directory(
        main_directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        main_directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Save the class labels
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    with open('class_labels.json', 'w') as f:
        json.dump(class_labels, f)

    # Load pre-trained ResNetV2-50 model without top classification layer
    base_model = ResNet50V2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Add classification layers on top of the base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save the model
    model_path = "resnet50v2_classifier.keras"
    model.save(model_path)
    return model_path





import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path):
    img_height, img_width = 100, 100  # Input size for ResNetV2
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (img_height, img_width))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array




def predict_classes():
    model = tf.keras.models.load_model("resnet50v2_classifier.keras")

    # Load class labels
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)

    # Parameters
    folder_path = "./faces"
    predicted_class_names = []

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, img_file)
            img_array = load_and_preprocess_image(img_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            prediction_accuracy = np.max(predictions, axis=1)[0]
            
            if prediction_accuracy >= 0.6:  # 60% threshold
                predicted_class_name = class_labels[str(predicted_class)]
            else:
                predicted_class_name = 'Non Registered Image'
            
            print(img_path, prediction_accuracy, predicted_class_name)
            predicted_class_names.append(predicted_class_name)
    
    for name in predicted_class_names:
        print(name)
    
    return set(predicted_class_names)



