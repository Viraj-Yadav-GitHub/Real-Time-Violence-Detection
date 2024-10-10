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
history = model.fit(X_train, y_train, epochs=2, callbacks=callbacks, validation_data=(X_test, y_test), batch_size=4)

# Save the trained model for future use
model.save('violence_detection_model.h5')
print('Model saved successfully.')

# Load the best weights
print('Restoring best weights...')
model.load_weights(checkpoint_filepath)
print('Weights restored successfully.')

# Real-time violence detection function
def preprocess_frame(frame):
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    return resized / 255.0  # Normalizing the image

def detect_humans(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Function to classify actions based on pose key points (for Punch, Kick, etc.)
def classify_action(pose_landmarks):
    if pose_landmarks:  # If pose landmarks are detected
        # Example logic based on the hand and foot positions
        left_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_foot = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        head = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Detect punch (right hand moves forward horizontally)
        if right_hand.x > 0.6 and left_hand.x < 0.6:  # Right hand moves to the right (example logic)
            return "Punch"
        
        # Detect kick (right foot moves up vertically)
        elif right_foot.y < 0.4:  # Right foot moves up (example logic)
            return "Kick"
        
        # Detect slap (right hand moves horizontally across the head height)
        elif abs(right_hand.y - head.y) < 0.1 and right_hand.x > 0.5:  # Hand near head, moving across
            return "Slap"
        
        # Detect elbow (elbow is raised and moves close to the head with an angular motion)
        elif abs(right_elbow.y - right_shoulder.y) < 0.1 and abs(right_elbow.x - head.x) < 0.1:
            return "Elbow"
        
        # You can add more logic for other actions like "slap", "elbow hit", etc.
        else:
            return "No Action"
    return None

# Function for real-time violence detection using webcam
def detect_violence_in_video(video_path, model):
    print("Starting video capture...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_preprocessed = preprocess_frame(frame)
        prediction = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]

        violence_text = "Violence Detected: No"
        if prediction > 0.5:
            violence_text = "Violence Detected: Yes"

        
         # Detect human actions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            action = classify_action(results.pose_landmarks)
            if action:
                cv2.putText(frame, f"Action Detected: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display violence detection
        cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Violence Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            

        detect_humans(frame)
        cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Violence Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load pre-trained model and run real-time detection
loaded_model = load_model('violence_detection_model.h5')
print('Model loaded from disk.')

# Start real-time detection using webcam
detect_violence_in_video(0, loaded_model)  # 0 for webcam input
