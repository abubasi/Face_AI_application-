from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import threading
import logging

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
HAAR_FILE = 'haarcascade_frontalface_default.xml'
DATASET = 'dataset'
CONFIDENCE_THRESHOLD = 80
WIDTH, HEIGHT = 130, 100

# Global variables
face_cascade = cv2.CascadeClassifier(HAAR_FILE)
model = cv2.face.LBPHFaceRecognizer_create()
webcam = None
names = {}
images, labels = [], []
capture_active = False
sub_data = None
path = None
count = 0

# --- Load Training Data ---
def load_training_data():
    global names, images, labels
    for (subdirs, dirs, files) in os.walk(DATASET):
        for subdir in dirs:
            id = len(names)
            names[id] = subdir  # Correctly map IDs to names
            subdir_path = os.path.join(DATASET, subdir)
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                image = cv2.imread(file_path, 0)
                if image is not None:
                    images.append(image)
                    labels.append(id)
    if images:
        model.train(np.array(images), np.array(labels))
        logging.info("Model trained successfully.")
    else:
        logging.warning("No training data available.")

load_training_data()

# --- Webcam Initialization ---
def initialize_webcam():
    global webcam
    if webcam is None or not webcam.isOpened():
        webcam = cv2.VideoCapture(0)
        logging.info("Webcam initialized.")

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/next')
def next_page():
    return render_template('index1.html')

@app.route('/initialize', methods=['POST'])
def initialize_dataset():
    global sub_data, path, count
    sub_data = request.form.get('name', '').strip()
    if not sub_data:
        return jsonify({"error": "Name cannot be empty"}), 400
    path = os.path.join(DATASET, sub_data)
    os.makedirs(path, exist_ok=True)
    count = 0
    return jsonify({"message": f"Dataset initialized for {sub_data}"}), 200

@app.route('/start', methods=['POST'])
def start_capture():
    global capture_active, count, path
    if not sub_data:
        return jsonify({"error": "User name not initialized"}), 400
    initialize_webcam()
    if not webcam.isOpened():
        return jsonify({"error": "Webcam not accessible"}), 500
    capture_active = True
    count = 0
    threading.Thread(target=process_frames).start()
    return jsonify({"message": "Capture started"}), 200

@app.route('/stop', methods=['POST'])
def stop_capture():
    global capture_active, webcam
    capture_active = False
    if webcam and webcam.isOpened():
        webcam.release()
        cv2.destroyAllWindows()
    return jsonify({"message": "Capture stopped"}), 200

@app.route('/status')
def capture_status():
    if count >= 50:
        return jsonify({"status": "limit_reached"}), 200
    return jsonify({"status": "capturing", "count": count}), 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Helper Functions ---
def process_frames():
    global count, capture_active, path
    while capture_active and count < 50:
        ret, frame = webcam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (WIDTH, HEIGHT))
            cv2.imwrite(f'{path}/{count + 1}.png', resized_face)
            count += 1
        if count >= 50:
            capture_active = False
            break

def generate_frames():
    global webcam
    initialize_webcam()
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (WIDTH, HEIGHT))
            prediction, confidence = model.predict(face_resize)
            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(prediction, "Unknown")
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# # --- Run the App ---
# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, Response, request, jsonify, render_template
# import cv2
# import os
# import threading
# import numpy as np
# import logging

# app = Flask(__name__)

# # Constants
# HAAR_FILE = 'haarcascade_frontalface_default.xml'
# DATASET = 'dataset'
# WIDTH, HEIGHT = 130, 100
# CONFIDENCE_THRESHOLD = 80
# FACE_CASCADE = cv2.CascadeClassifier(HAAR_FILE)

# # Initialize variables
# webcam = None
# capture_active = False
# count = 0
# sub_data = None
# path = None

# # Variables for face recognition
# model = cv2.face.LBPHFaceRecognizer_create()
# names = {}
# images, labels = [], []

# # Logging configuration
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# # Load dataset and train the model
# def load_and_train_model():
#     global model, images, labels, names
#     for (subdirs, dirs, files) in os.walk(DATASET):
#         for subdir in dirs:
#             id = len(names)
#             names[id] = subdir
#             subdir_path = os.path.join(DATASET, subdir)
#             for filename in os.listdir(subdir_path):
#                 path = os.path.join(subdir_path, filename)
#                 image = cv2.imread(path, 0)  # Load as grayscale
#                 if image is not None:
#                     images.append(image)
#                     labels.append(id)

#     if images:
#         images, labels = np.array(images), np.array(labels)
#         model.train(images, labels)
#         logging.info("Model trained successfully with %d images.", len(images))
#     else:
#         logging.warning("No training data found. Ensure dataset has images.")

# # Initialize the webcam
# def initialize_webcam():
#     global webcam
#     if webcam is None or not webcam.isOpened():
#         webcam = cv2.VideoCapture(0)
#         logging.info("Webcam initialized.")

# # Generate video frames with face detection
# def generate_frames():
#     global webcam, model, names

#     # Ensure the model is trained before predicting
#     if len(images) == 0 or len(labels) == 0:
#         yield (b'--frame\r\n'
#                b'Content-Type: text/plain\r\n\r\n'
#                b'Model not trained. Ensure dataset is properly initialized.\r\n')
#         return

#     initialize_webcam()

#     while True:
#         success, frame = webcam.read()
#         if not success:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             face = gray[y:y + h, x:x + w]
#             face_resize = cv2.resize(face, (width, height))
#             prediction = model.predict(face_resize)

#             if prediction[1] < confidence_threshold:
#                 name = names.get(prediction[0], "Unknown")
#                 color = (0, 255, 0)  # Green for known faces
#             else:
#                 name = "Unknown"
#                 color = (0, 0, 255)  # Red for unknown faces

#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# # Start capturing images
# def process_frames():
#     global webcam, count, capture_active, path

#     while capture_active and count < 50:
#         ret, frame = webcam.read()
#         if not ret:
#             continue

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             face = gray[y:y + h, x:x + w]
#             resized_face = cv2.resize(face, (WIDTH, HEIGHT))
#             cv2.imwrite(f'{path}/{count + 1}.png', resized_face)
#             count += 1

#         if count >= 50:
#             capture_active = False
#             break

# # Route for the home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to the next page
# @app.route('/next')
# def next_page():
#     return render_template('index1.html')

# # Route to initialize the user dataset
# @app.route('/initialize', methods=['POST'])
# def initialize():
#     global sub_data, path, count

#     sub_data = request.form.get('name', '').strip()
#     if not sub_data:
#         return jsonify({"error": "Name cannot be empty"}), 400

#     path = os.path.join(DATASET, sub_data)
#     if not os.path.exists(path):
#         os.makedirs(path)

#     count = 0
#     return jsonify({"message": f"Dataset initialized for {sub_data}"}), 200

# # Route to start capturing images
# @app.route('/start', methods=['POST'])
# def start_capture():
#     global webcam, capture_active, count

#     if not sub_data:
#         return jsonify({"error": "User name not initialized"}), 400

#     # Start webcam
#     webcam = cv2.VideoCapture(0)
#     if not webcam.isOpened():
#         return jsonify({"error": "Webcam not accessible"}), 500

#     capture_active = True
#     count = 0

#     # Start a separate thread for capturing images
#     threading.Thread(target=process_frames).start()
#     return jsonify({"message": "Capture started"}), 200

# # Route to stop capturing images
# @app.route('/stop', methods=['POST'])
# def stop_capture():
#     global capture_active, webcam
#     capture_active = False

#     if webcam:
#         webcam.release()
#         cv2.destroyAllWindows()
#     return jsonify({"message": "Capture stopped"}), 200

# # Route to stream video feed with face detection
# @app.route('/stream')
# def stream():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Route to check capture status
# @app.route('/status')
# def status():
#     global count
#     if count >= 50:
#         return jsonify({"status": "limit_reached"}), 200
#     return jsonify({"status": "capturing", "count": count}), 200

# if __name__ == '__main__':
#     load_and_train_model()  # Train the model at the start
#     app.run(debug=True, use_reloader=False)
