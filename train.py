import tensorflow as tf  # TensorFlow for deep learning
from keras.api.models import Sequential  # Sequential API for building models
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense  # CNN layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Define paths where the dataset is stored
dataset_path = 'images/'  # Path to your dataset folder

# Data augmentation and preprocessing
data_gen = ImageDataGenerator(
    rescale=1./255,    # Normalize pixel values to [0,1] improving neural network performance
    validation_split=0.2  # Reserve 20% for validation and rest for training
)

# Load training and validation data
train_data = data_gen.flow_from_directory(
    dataset_path,  # Path to dataset folder
    target_size=(300, 300),  # Resize all images to 300x300 pixels
    batch_size=32,  # Number of images processed at once
    class_mode='categorical',  # Multi-class classification (one-hot encoded labels)
    subset='training'  # This loads the 80% training subset
)

# Load and preprocess validation data
val_data = data_gen.flow_from_directory(
    dataset_path, # Path to dataset folder
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    subset='validation' #20% validation wala
)

# Define the CNN model
model = Sequential([
    #1st layer:extract features from the image
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D((2, 2)),  # Reduces image size by 2x and retains important features

    #2nd: extracts more complex features
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    #3rd : deep patterns
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(), #convert 2d map into 1d array for dense layer

    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # Output layer matches the number of classes
])
# Compile the model with these 3
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Train the model using training & valid
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10 #no of times model see entire dataset
)
# Save the model
model.save('hand_gesture_model.h5')
print("Model trained and saved as 'hand_gesture_model.h5'")  #confirmation msg
