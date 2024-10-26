from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_pymongo import PyMongo
import os
import cv2
import numpy as np
import threading
from keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import json
import mediapipe as mp
import pygame  # Import pygame


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/violence_detection"  # Your database name
mongo = PyMongo(app)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

pygame.mixer.init()
beep_sound = pygame.mixer.Sound('beep.mp3')  

# Load your pretrained Keras model (replace with the correct path to your model)
model = load_model('violence_detection_model.h5')  # Update with the correct model path
print('Model loaded from disk.')

# Global variables to control detection thread and webcam capture
detection_running = False
video_capture = None

# Image size used in the model
IMG_SIZE = 128

# Initialize database collections if not present
def init_db():
    mongo.db.users.create_index('username', unique=True)

def preprocess_frame(frame):
    """Preprocess the frame before passing to the model."""
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    return resized / 255.0  # Normalizing the image

def detect_humans(frame):
    """Detect humans in the frame and draw rectangles around them."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans, _ = hog.detectMultiScale(gray_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def save_frame_to_device(frame):
    """Save the detected frame to a folder for all users."""
    user_folder = 'captured_frames'  # Folder for captured frames

    # Create directory if it doesn't exist
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Generate a unique filename with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(user_folder, f"frame_{timestamp}.jpg")

    # Save the frame to the file
    cv2.imwrite(filename, frame)
    print(f"Screenshot taken and saved: {filename}")


# Actio detection


import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Assuming 'pose' is defined as:
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

def detect_actions(frame, previous_landmarks):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    action_type = "None"
    action_detected = False

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        if previous_landmarks:
            # Calculate velocities of wrists and ankles
            left_wrist_velocity = abs(left_wrist.x - previous_landmarks[0]) + abs(left_wrist.y - previous_landmarks[1])
            right_wrist_velocity = abs(right_wrist.x - previous_landmarks[2]) + abs(right_wrist.y - previous_landmarks[3])
            left_ankle_velocity = abs(left_ankle.y - previous_landmarks[4])
            right_ankle_velocity = abs(right_ankle.y - previous_landmarks[5])
            velocity_threshold = 0.05  # Lower threshold for more sensitivity

            # Detect Slapping and Boxing
            if left_wrist_velocity > velocity_threshold or right_wrist_velocity > velocity_threshold:
                if abs(left_wrist.y - left_elbow.y) < 0.15 or abs(right_wrist.y - right_elbow.y) < 0.15:
                    if abs(left_wrist.y - left_elbow.y) < 0.05 or abs(right_wrist.y - right_elbow.y) < 0.05:
                        action_type = "Slapping"
                    else:
                        action_type = "Boxing"
                    action_detected = True
            
            # Detect Punching
            if (left_wrist_velocity > velocity_threshold and left_wrist.x > left_elbow.x) or \
               (right_wrist_velocity > velocity_threshold and right_wrist.x > right_elbow.x):
                action_type = "Punching"
                action_detected = True

            # Detect Kicking
            if left_ankle_velocity > velocity_threshold or right_ankle_velocity > velocity_threshold:
                if left_ankle.y < left_wrist.y or right_ankle.y < right_wrist.y:
                    action_type = "Kicking"
                    action_detected = True

            # Detect Pushing
            if (left_wrist_velocity > velocity_threshold and left_wrist.x > left_shoulder.x) or \
               (right_wrist_velocity > velocity_threshold and right_wrist.x > right_shoulder.x):
                action_type = "Pushing"
                action_detected = True

            # Detect Elbowing
            if (left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y) and \
               (abs(left_elbow.y - right_elbow.y) < 0.1):
                action_type = "Elbowing"
                action_detected = True

            # Detect Headbutting (assuming the head is positioned at a specific landmark)
            head_landmark = landmarks[mp_pose.PoseLandmark.NOSE.value]
            if (left_wrist.y < head_landmark.y or right_wrist.y < head_landmark.y) and \
               (abs(head_landmark.y - left_shoulder.y) < 0.2 or abs(head_landmark.y - right_shoulder.y) < 0.2):
                action_type = "Headbutting"
                action_detected = True

            # Detect Grappling (detected by close proximity of wrists)
            if abs(left_wrist.x - right_wrist.x) < 0.1 and abs(left_wrist.y - right_wrist.y) < 0.1:
                action_type = "Grappling"
                action_detected = True
            
            # Detect Throwing (if the wrist moves downwards suddenly)
            if (left_wrist_velocity > velocity_threshold and left_wrist.y < left_shoulder.y) or \
               (right_wrist_velocity > velocity_threshold and right_wrist.y < right_shoulder.y):
                action_type = "Throwing"
                action_detected = True

        previous_landmarks = [left_wrist.x, left_wrist.y, right_wrist.x, right_wrist.y, 
                              left_ankle.y, right_ankle.y, 
                              left_elbow.y, right_elbow.y]
    else:
        previous_landmarks = None

    return action_type, action_detected, previous_landmarks




import cv2
import numpy as np
import time

# Initialize variables
detection_running = False

# Initialize HOGDescriptor for human detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_violence_in_webcam(model):
    """Function to capture frames from webcam and detect violence with enhanced features and human bounding box."""
    cap = cv2.VideoCapture(0)
    previous_landmarks = None
    red_text_end_time = 0
    violence_detected = False
    alert_message = "AuthorityAlert: Negative"
    text_color = (0, 255, 0)  # Default green color for no violence

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed...")

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess frame and make prediction
        frame_preprocessed = preprocess_frame(frame)
        prediction = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]

        # Human detection
        humans, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

        # Draw bounding boxes around detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for human detection

        # Action detection logic
        action_type, action_detected, previous_landmarks = detect_actions(frame, previous_landmarks)
        current_time = time.time()

        # Detect violence and update message/color if prediction score is above the threshold
        if action_detected and prediction > 0.5:
            violence_detected = True
            alert_message = "AuthorityAlert: Positive"
            text_color = (0, 0, 255)  # Red color for violence
            save_frame_to_device(frame)
            beep_sound.play()  # Assuming `play_alert_sound` is defined to play a beep or alert sound
            red_text_end_time = current_time + 2  # Show red text for 2 seconds

        # Display red text during alert period
        if current_time < red_text_end_time:
            cv2.putText(frame, "Violence Detected: Yes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, f"Action Detected: {action_type}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, alert_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        else:
            # Reset to no violence detected
            violence_detected = False
            alert_message = "AuthorityAlert: Negative"
            text_color = (0, 255, 0)  # Green color for no violence
            cv2.putText(frame, "Violence Detected: No", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, "Action Detected: None", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(frame, alert_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Display the frame with current alerts and bounding boxes
        cv2.imshow("Violence Detection", frame)

        # Press 'q' to exit the detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Assuming `preprocess_frame`, `detect_actions`, `save_frame_to_device`, and `play_alert_sound` are defined elsewhere in the code



# Modify the start_detection function to call detect_violence_in_webcam
def start_detection():
    """Start capturing video and detecting violence."""
    global detection_running
    detection_running = True
    detect_violence_in_webcam(model)  # Call detect_violence_in_webcam with the model


@app.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['username'] = username  # Store username in session
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user signup."""
    if request.method == 'POST':
        name = request.form.get('name')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        mobile_number = request.form.get('mobile_number')
        email = request.form.get('email')
        place = request.form.get('place')

        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)

        try:
            # Insert new user into the database
            mongo.db.users.insert_one({
                'name': name,
                'username': username,
                'password': hashed_password,
                'mobile_number': mobile_number,
                'email': email,
                'place': place
            })
            flash('Signup successful!')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error occurred: {str(e)}. Please try again.')
            return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/profile')
def profile():
    """Render the profile page for logged-in users."""
    if 'logged_in' in session:
        user = mongo.db.users.find_one({'username': session['username']})
        return render_template('profile.html', user=user)
    flash('Please log in to access your profile.')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    """Log out the user and clear the session."""
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/real_time')
def real_time():
    """Render the real-time detection page."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('real_time.html')

@app.route('/pre_recorded')
def pre_recorded():
    """Render the pre-recorded video detection page."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('pre_recorded.html')

@app.route('/start_detection', methods=['POST'])
def start_detection_endpoint():
    """Start the detection thread if it's not already running."""
    global detection_running
    if not detection_running:
        detection_thread = threading.Thread(target=start_detection)
        detection_thread.start()
        return jsonify({"message": "Detection started!"}), 200
    else:
        return jsonify({"message": "Detection is already running!"}), 400

@app.route('/stop_detection', methods=['POST'])
def stop_detection_endpoint():
    """Stop the detection thread."""
    global detection_running
    detection_running = False
    return jsonify({"message": "Detection stopped!"}), 200

@app.route('/get_captured_frames', methods=['GET'])
def get_captured_frames():
    """Return a list of captured frames with timestamps."""
    user_folder = 'captured_frames'
    frames = []

    if os.path.exists(user_folder):
        # Get the last 20 captured frames sorted by time
        captured_files = sorted([f for f in os.listdir(user_folder) if f.endswith('.jpg')],
                                key=lambda x: os.path.getmtime(os.path.join(user_folder, x)),
                                reverse=True)[:21]
        for filename in captured_files:
            timestamp = os.path.getmtime(os.path.join(user_folder, filename))
            frames.append({
                "image": f"/captured_frames/{filename}",
                "timestamp": timestamp
            })

    return jsonify({"frames": frames})

@app.route('/captured_frames/<filename>')
def send_captured_frame(filename):
    """Serve the captured frames."""
    return send_from_directory('captured_frames', filename)

@app.route('/check_login')
def check_login():
    """Check if the user is logged in."""
    return ('', 200) if session.get('logged_in') else ('', 401)



#pre-recorded

@app.route('/detect_video', methods=['POST'])
def detect_video():
    """Detect violence in the uploaded video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded.'}), 400

    video_file = request.files['video']
    
    # Save the video to a temporary location
    temp_video_path = os.path.join('uploads', video_file.filename)
    video_file.save(temp_video_path)

    # Initialize detection results
    results = {
        'frames': [],
        'violence_detected': False
    }

    # Open the video and process it frame by frame
    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        return jsonify({'error': 'Could not open video file.'}), 400

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for prediction
        frame_preprocessed = preprocess_frame(frame)  # Ensure this function is defined
        prediction = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]

        # Check if violence is detected
        if prediction > 0.5:
            results['violence_detected'] = True

        # Save frame result for display
        results['frames'].append({
            'violence': 'Yes' if prediction > 0.5 else 'No'
        })

    cap.release()  # Release the video capture object
    os.remove(temp_video_path)  # Remove the temporary video file

    # Debugging line to print detection results
    print("Detection Results:", results)  
    
    # Return the results as JSON
    return jsonify(results)

if __name__ == '__main__':
    init_db()  # Initialize the database on startup
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
