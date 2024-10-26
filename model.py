#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import imgaug.augmenters as iaa
import datetime
import mediapipe as mp

# Platform details
print(f"Platform: {os.name}")

# Seed for reproducibility
tf.random.set_seed(73)

# Set up directories
MyDrive = '/kaggle/working'
PROJECT_DIR = './Downloads/archive'

IMG_SIZE = 128
ColorChannels = 3
VideoDataDir = '/Real Life Violence Dataset'
CLASSES = ["NonViolence", "Violence"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to ensure directory exists
def resolve_dir(Dir):
    if not os.path.exists(Dir):
        os.mkdir(Dir)

# Function to reset a directory by removing all files within it
def reset_path(Dir):
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    else:
        os.system('rm -f {}/*'.format(Dir))

# Convert video frames to image frames
def video_to_frames(video):
    vidcap = cv2.VideoCapture(video)
    ImageFrames = []
    
    while vidcap.isOpened():
        ID = vidcap.get(1)
        success, image = vidcap.read()
        
        if success:
            if (ID % 7 == 0):  # skip frames to avoid duplication
                flip = iaa.Fliplr(1.0)
                zoom = iaa.Affine(scale=1.3)
                random_brightness = iaa.Multiply((1, 1.3))
                rotate = iaa.Affine(rotate=(-25, 25))
                
                image_aug = flip(image=image)
                image_aug = random_brightness(image=image_aug)
                image_aug = zoom(image=image_aug)
                image_aug = rotate(image=image_aug)
                
                rgb_img = cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
                ImageFrames.append(resized)
        else:
            break
    
    vidcap.release()
    return ImageFrames


import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the violence detection model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Preprocessing function: resize and normalize the frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    return frame_normalized

# Function to detect humans using HOG descriptor
def detect_humans(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    for (x, y, w, h) in humans:
        # Draw bounding box around each detected human
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Violence detection function with bounding boxes and alert
def detect_violence_in_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    violence_detected = False
    alert_message = "PoliceAlert: Negative"
    violence_text = "Violence Detected: No"

    print(f"Processing video: {video_path.split('/')[-1]}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for violence detection
        processed_frame = preprocess_frame(frame)
        
        # Predict violence
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        
        # Update alert message based on prediction
        if prediction > 0.3:
            violence_detected = True
            alert_message = "PoliceAlert: Positive"
            violence_text = "Violence Detected: Yes"
        else:
            alert_message = "PoliceAlert: Negative"
            violence_text = "Violence Detected: No"
        
        # Detect humans in the frame and draw bounding boxes
        detect_humans(frame)

        # Display the violence detection status at the top of the video
        cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, alert_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video frame by frame
        cv2.imshow('Video', frame)
        
        # Add a small delay and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Automatically close the window after the video ends

    # Final alert message after video processing
    if violence_detected:
        print("Violence detected in the video. PoliceAlert: Positive")
    else:
        print("No violence detected in the video. PoliceAlert: Negative")

# Path to the video file
video_path = 'D:/Real Life Violence Dataset/Violence/V_19.mp4'

# Run the violence detection on the specific video
detect_violence_in_video(video_path, model)




import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the violence detection model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Preprocessing function: resize and normalize the frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    return frame_normalized

# Function to detect humans using HOG descriptor
def detect_humans(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect humans in the frame
    humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    for (x, y, w, h) in humans:
        # Draw bounding box around each detected human
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Violence detection function with bounding boxes and alert
def detect_violence_in_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    violence_detected = False
    alert_message = "PoliceAlert: Negative"
    violence_text = "Violence Detected: No"

    print(f"Processing video: {video_path.split('/')[-1]}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for violence detection
        processed_frame = preprocess_frame(frame)
        
        # Predict violence
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        prediction_score = prediction[0][0]  # Get the actual prediction score
        
        # Print the prediction score for debugging
        print(f"Prediction score: {prediction_score}")
        
        # Update alert message based on prediction score
        if prediction_score > 0.6:  # You may try lowering this threshold
            violence_detected = True
            alert_message = "PoliceAlert: Positive"
            violence_text = "Violence Detected: Yes"
        else:
            alert_message = "PoliceAlert: Negative"
            violence_text = "Violence Detected: No"
        
        # Detect humans in the frame and draw bounding boxes
        detect_humans(frame)

        # Display the violence detection status at the top of the video
        cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, alert_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the video frame by frame
        cv2.imshow('Video', frame)
        
        # Add a small delay and break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Automatically close the window after the video ends

    # Final alert message after video processing
    if violence_detected:
        print("Violence detected in the video. PoliceAlert: Positive")
    else:
        print("No violence detected in the video. PoliceAlert: Negative")

# Path to the video file
video_path = 'D:/Real Life Violence Dataset/NonViolence/NV_19.mp4'

# Run the violence detection on the specific video
detect_violence_in_video(video_path, model)




# Load and process video data
print(f"We have \n{len(os.listdir(os.path.join(VideoDataDir, 'Violence')))} Violence videos")
print(f"{len(os.listdir(os.path.join(VideoDataDir, 'NonViolence')))} NonViolence videos")

X_original = []
y_original = []

for category in CLASSES:
    path = os.path.join(VideoDataDir, category)
    class_num = CLASSES.index(category)
    
    for i, video in enumerate(tqdm(os.listdir(path)[:350])):  # Limit to 350 for memory efficiency
        frames = video_to_frames(os.path.join(path, video))
        for frame in frames:
            X_original.append(frame)
            y_original.append(class_num)

# Split data into training and test sets using stratified sampling
X_original = np.array(X_original)
y_original = np.array(y_original)

stratified_sample = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=73)
for train_index, test_index in stratified_sample.split(X_original, y_original):
    X_train, X_test = X_original[train_index], X_original[test_index]
    y_train, y_test = y_original[train_index], y_original[test_index]

# Define MobileNetV2-based model architecture
def load_layers():
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, ColorChannels))
    baseModel = tf.keras.applications.MobileNetV2(pooling='avg', include_top=False, input_tensor=input_tensor)
    
    headModel = baseModel.output
    headModel = Dense(1, activation="sigmoid")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    print("Compiling model...")
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model

# Load model layers and compile
model = load_layers()
model.summary()

# Callbacks for training
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.999:
            print("\nLimits Reached! Stopping training.")
            self.model.stop_training = True

# Learning rate scheduling function
def lrfn(epoch):
    max_lr = 0.00005
    start_lr = 0.00001
    exp_decay = 0.8
    rampup_epochs = 5
    sustain_epochs = 0
    min_lr = 0.00001

    if epoch < rampup_epochs:
        return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
    elif epoch < rampup_epochs + sustain_epochs:
        return max_lr
    else:
        return (max_lr - min_lr) * exp_decay ** (epoch - rampup_epochs - sustain_epochs) + min_lr

end_callback = myCallback()
lr_callback = LearningRateScheduler(lrfn, verbose=False)

early_stopping = EarlyStopping(patience=3, monitor='val_loss', mode='min', restore_best_weights=True)
lr_plat = ReduceLROnPlateau(patience=2, mode='min')

# TensorBoard and checkpoint setup
tensorboard_log_dir = "logs/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

checkpoint_filepath = "ModelWeights.weights.h5"
model_checkpoints = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

callbacks = [end_callback, lr_callback, model_checkpoints, tensorboard_callback, early_stopping, lr_plat]

# Train the model
print('Training head...')
history = model.fit(X_train, y_train, epochs=70, callbacks=callbacks, validation_data=(X_test, y_test), batch_size=4)

# Save the trained model for future use
model.save('violence_detection_model.h5')
print('Model saved successfully.')

# Load the best weights
print('Restoring best weights...')
model.load_weights(checkpoint_filepath)
print('Weights restored successfully.')

# Load pre-trained model and run real-time detection
loaded_model = load_model('violence_detection_model.h5')
print('Model loaded from disk.')














# # Real-time violence detection function
# def preprocess_frame(frame):
#     rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
#     return resized / 255.0  # Normalizing the image

# def detect_humans(frame):
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
#     for (x, y, w, h) in humans:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# # Function to classify actions based on pose key points (for Punch, Kick, etc.)
# def classify_action(pose_landmarks):
#     if pose_landmarks:  # If pose landmarks are detected
#         # Example logic based on the hand and foot positions
#         left_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
#         right_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
#         left_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
#         right_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
#         head = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
#         left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
#         right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
#         left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

#         # Detect punch (right hand moves forward horizontally)
#         if right_hand.x > 0.6 and left_hand.x < 0.6:  # Right hand moves to the right (example logic)
#             return "Punch"
        
#         # Detect kick (right foot moves up vertically)
#         elif right_foot.y < 0.4:  # Right foot moves up (example logic)
#             return "Kick"
        
#         # Detect slap (right hand moves horizontally across the head height)
#         elif abs(right_hand.y - head.y) < 0.1 and right_hand.x > 0.5:  # Hand near head, moving across
#             return "Slap"
        
#         # Detect elbow (elbow is raised and moves close to the head with an angular motion)
#         elif abs(right_elbow.y - right_shoulder.y) < 0.1 and abs(right_elbow.x - head.x) < 0.1:
#             return "Elbow"
        
#         # You can add more logic for other actions like "slap", "elbow hit", etc.
#         else:
#             return "No Action"
#     return None

# # Function for real-time violence detection using webcam
# def detect_violence_in_video(video_path, model):
#     print("Starting video capture...")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         frame_preprocessed = preprocess_frame(frame)
#         prediction = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]

#         violence_text = "Violence Detected: No"
#         if prediction > 0.5:
#             violence_text = "Violence Detected: Yes"

        
#          # Detect human actions
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frame_rgb)
        
#         if results.pose_landmarks:
#             action = classify_action(results.pose_landmarks)
#             if action:
#                 cv2.putText(frame, f"Action Detected: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

#         # Display violence detection
#         cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Violence Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            

#         detect_humans(frame)
#         cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Violence Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



# Start real-time detection using webcam
# detect_violence_in_video(0, loaded_model)  # 0 for webcam input
