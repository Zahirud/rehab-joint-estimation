import cv2
import math
import mediapipe as mp
from flask import Flask, render_template, Response, request, redirect, url_for
from datetime import datetime
from playsound import playsound

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# MediaPipe Pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# OpenCV Video Capture (Default webcam)
cap = cv2.VideoCapture(0)

# State to track current menu
current_menu = {"part_name": None, "mode": None, "target_angle": None}
current_target_index = 0
reached_target = False

# Target angles dictionary
target_angles = []

# Calculate angle between three points
def calculate_angle(A, B, C):
    AB = [A[0] - B[0], A[1] - B[1]]
    BC = [C[0] - B[0], C[1] - B[1]]
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    try:
        angle = math.degrees(math.acos(dot_product / (magnitude_AB * magnitude_BC)))
    except ZeroDivisionError:
        angle = 0
    return angle


def check_target_angle(angle):
    global reached_target
    for activity in activity_log:
        if not activity["reached"] and abs(angle - activity["target_angle"]) <= 5:
            activity["reached"] = True
            activity["reached_at"] = datetime.now()
            activity["duration"] = (activity["reached_at"] - activity["submitted_at"]).total_seconds()
            reached_target = True
            print(f"Target {activity['target_angle']}° reached!")  # Debugging log


# Check if the angle is close to the target angle
def check_target_angle(angle):
    global current_target_index, reached_target
    if current_menu["target_angle"] is not None:
        target_angle = current_menu["target_angle"]
        if not reached_target and abs(angle - target_angle) <= 2:
            reached_target = True
            update_activity_status(target_angle, reached=True)



def generate_frames(selected_part, target_angle):
    global reached_target
    while True:
        if cap is None or not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Determine the relevant landmarks based on selected part and mode
            if selected_part == 'Right_Shoulder' and current_menu["mode"] in ["extension", "flexion"]:
                relevant_indices = [23, 11, 13]
            elif selected_part == 'Left_Shoulder' and current_menu["mode"] in ["extension", "flexion"]:
                relevant_indices = [24, 12, 14]
            else:
                relevant_indices = {
                    'Right_Elbow': [11, 13, 15],
                    'Left_Elbow': [12, 14, 16],
                    'Right_Shoulder': [12, 11, 13],  # Default
                    'Left_Shoulder': [11, 12, 14],  # Default
                }.get(selected_part, [])

            coords = [
                (int(landmarks[idx].x * frame.shape[1]), int(landmarks[idx].y * frame.shape[0]))
                for idx in relevant_indices
            ]

            for i, coord in enumerate(coords):
                cv2.circle(frame, coord, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame, coords[i - 1], coord, (255, 255, 255), 2)

            if len(coords) == 3:
                angle = calculate_angle(coords[0], coords[1], coords[2])
                check_target_angle(angle)

                if not reached_target and abs(angle - target_angle) <= 1:
                    reached_target = True
                    print(f"Target {target_angle}° reached!")  # Debugging log

            if selected_part == 'Right_Elbow':
                right_angle = calculate_angle(coords[0], coords[1], coords[2])
                cv2.putText(frame, f'Right Elbow Angle: {int(right_angle)}', (coords[1][0] + 10, coords[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif selected_part == 'Left_Elbow':
                left_angle = calculate_angle(coords[0], coords[1], coords[2])
                cv2.putText(frame, f'Left Elbow Angle: {int(left_angle)}', (coords[1][0] + 10, coords[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif selected_part == 'Right_Shoulder':
                Right_Shoulder_angle = calculate_angle(coords[0], coords[1], coords[2])
                cv2.putText(frame, f'Right Shoulder Angle: {int(Right_Shoulder_angle)}', (coords[1][0] + 10, coords[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            elif selected_part == 'Left_Shoulder':
                Left_Shoulder_angle = calculate_angle(coords[0], coords[1], coords[2])
                cv2.putText(frame, f'Left Shoulder Angle: {int(Left_Shoulder_angle)}', (coords[1][0] + 10, coords[1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if reached_target:
            # Display "Target Reached" message
            cv2.putText(frame, "Target Reached!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

            # Add the button for stopping the camera (it's just an example, you can't show HTML in OpenCV)
            cv2.putText(frame, "Congratulation!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Convert the frame to JPEG to send to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")  # Debug log
            break

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Initialize activity log
activity_log = []

# Route to display activities
@app.route('/activities')
def activities():
    """Display the logged user activities."""
    return render_template('activities.html', activities=activity_log)


# Initialize activity log
activity_log = []

# Log activity with 'reached' status
def log_activity(part_name, mode, target_angle, reached=False):
    activity_log.append({
        "part_name": part_name,
        "mode": mode,
        "target_angle": target_angle,
        "reached": reached,
        "submitted_at": datetime.now(),  # Log date and time
        "reached_at": None,  # To store when the target was reached
        "duration": None     # To calculate the time taken to reach
    })

# Update 'reached' status in activity log
def update_activity_status(target_angle, reached):
    for activity in activity_log:
        if activity["target_angle"] == target_angle and not activity["reached"]:
            activity["reached"] = reached
            activity["reached_at"] = datetime.now()  # Log when reached
            # Calculate duration in seconds
            activity["duration"] = (activity["reached_at"] - activity["submitted_at"]).total_seconds()
            break



# Route to the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to body part submenu
@app.route('/submenu/<part_name>')
def submenu(part_name):
    current_menu["part_name"] = part_name
    return render_template('submenu.html', part_name=part_name)

@app.route('/target_input/<part_name>/<mode>', methods=['GET', 'POST'])
def target_input(part_name, mode):
    global reached_target
    if request.method == 'POST':
        target_angle = request.form.get('target_angle', type=int)
        if target_angle is None or not (0 <= target_angle <= 360):
            error_message = "The angle must be between 0 and 360 degrees."
            return render_template('target_input.html', part_name=part_name, mode=mode, error=error_message)

        current_menu.update({"mode": mode, "target_angle": target_angle})
        reached_target = False  # Reset for the new target
        log_activity(part_name, mode, target_angle)  # Log new activity

        return redirect(url_for('video_feed', selected_part=part_name, target_angle=target_angle))

    return render_template('target_input.html', part_name=part_name, mode=mode)

@app.route('/video_feed/<selected_part>/<int:target_angle>')
def video_feed(selected_part, target_angle):
    global cap, activity_start_time

    # Record the activity start time
    activity_start_time = datetime.now()

    # Release the camera if it is already initialized
    if cap is not None and cap.isOpened():
        cap.release()

    # Reinitialize the camera
    cap = cv2.VideoCapture(0)

    # Generate the video feed frames
    return Response(generate_frames(selected_part, target_angle), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/menu')
def menu():
    return render_template('menu.html')


# Route for instructions page
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

# Route to stop the camera
@app.route('/stop_camera')
def stop_camera():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()  # Release the webcam resource
    return redirect(url_for('index'))  # Redirect to the main menu

if __name__ == '__main__':
    app.run(debug=True, threaded=True)