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

def start_detection():
    """Start capturing video and detecting violence in real-time."""
    global detection_running
    detection_running = True
    cap = cv2.VideoCapture(0)  # Start capturing video from the webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        detection_running = False
        return

    print("Starting video capture...")

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_preprocessed = preprocess_frame(frame)
        prediction = model.predict(np.expand_dims(frame_preprocessed, axis=0))[0][0]

        violence_text = "Violence Detected: No"
        if prediction > 0.7:
            violence_text = "Violence Detected: Yes"
            # beep_sound.play()
            save_frame_to_device(frame)  # Save the frame when violence is detected

        detect_humans(frame)  # Detect humans and draw rectangles

          # Detect human actions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            action = classify_action(results.pose_landmarks)
            if action:
                cv2.putText(frame, f"Action Detected: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.putText(frame, violence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Violence Detection", frame)

        # Check if 'q' is pressed to quit or detection_running is False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detection_running = False

    cap.release()  # Release the webcam when done
    cv2.destroyAllWindows()  # Close all OpenCV windows

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
