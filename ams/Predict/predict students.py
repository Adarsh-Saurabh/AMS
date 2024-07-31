import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# def predict_student():
# Load the saved model
# model = tf.keras.models.load_model(r"D:\adarsh ka project\Attedence Management System\ams\resnet50v2_classifier.h5")

# # Parameters
# img_height, img_width = 100, 100  # Input size for ResNetV2

# # Function to load and preprocess images
# def load_and_preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(img_height, img_width))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0
#     return img_array

# # Function to predict the class of images in a folder
# def predict_classes(folder_path):
#     class_names = set()
#     for img_file in os.listdir(folder_path):
#         if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             img_path = os.path.join(folder_path, img_file)
#             img_array = load_and_preprocess_image(img_path)
#             predictions = model.predict(img_array)
#             predicted_class = np.argmax(predictions, axis=1)[0]
#             class_names.add(predicted_class)
#     return class_names

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
