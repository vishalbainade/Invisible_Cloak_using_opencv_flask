from flask import Flask, render_template, request, send_file, redirect, url_for
import cv2
import numpy as np
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
application = app
# Parameters for red color detection (default values)
lower_hue = 0
upper_hue = 10
saturation = 120
value = 70

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global lower_hue, upper_hue, saturation, value

    # Get color detection parameters from the form
    lower_hue = int(request.form['lower_hue'])
    upper_hue = int(request.form['upper_hue'])
    saturation = int(request.form['saturation'])
    value = int(request.form['value'])

    # Upload video file
    uploaded_file = request.files['file']
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        video_path = os.path.join(tempfile.gettempdir(), filename)
        uploaded_file.save(video_path)
        
        # Process video
        output_path = process_video(video_path)
        
        return send_file(output_path, as_attachment=True, download_name="output_invisible_cloak.mp4", mimetype='video/mp4')

    return redirect(url_for('index'))

def process_video(video_path):
    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)
    
    # Capture the background from the first few frames
    ret, background = cap.read()
    if ret:
        background = np.flip(background, axis=1)
    
    # Prepare to save the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(tempfile.gettempdir(), 'output_invisible_cloak.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect red color
        lower_red = np.array([lower_hue, saturation, value])
        upper_red = np.array([upper_hue, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        lower_red = np.array([170, saturation, value])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        mask = mask1 + mask2
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # Replace red pixels with background pixels
        frame[np.where(mask == 255)] = background[np.where(mask == 255)]
        
        # Save frame to output video
        out.write(frame)

    cap.release()
    out.release()

    return output_path

if __name__ == "__main__":
    app.run(debug=True)
