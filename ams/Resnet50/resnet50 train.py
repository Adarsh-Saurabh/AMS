
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def ml_function():
    main_directory = "./people"

    # Parameters
    img_height, img_width = 100, 100  # Input size for ResNetV2
    batch_size = 32
    epochs = 10
    num_classes = len(os.listdir(main_directory))  # Number of class folders

    # Create ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )

    # Generate batches of augmented data from the directories
    train_generator = train_datagen.flow_from_directory(
        main_directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        main_directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Load pre-trained ResNetV2-50 model without top classification layer
    base_model = ResNet50V2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Add classification layers on top of the base model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
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
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the model
    model_path = "resnet50v2_classifier.h5"
    model.save(model_path)
    return model_path