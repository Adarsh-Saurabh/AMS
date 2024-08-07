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

# Function to predict the class of a single image
def predict_image_class(img_path, model_path=r"D:\adarsh ka project\Attedence Management System\ams\resnet50v2_classifier.h5"):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)
    
    # Predict the class of the image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class

# Example usage
folder_path = r"D:\adarsh ka project\Attedence Management System\ams\faces"

predicted_classes = predict_image_class(r"D:\adarsh ka project\Attedence Management System\ams\faces\face_0_1.jpg", r"D:\adarsh ka project\Attedence Management System\ams\resnet50v2_classifier.h5" )
print(predicted_classes)
