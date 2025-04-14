from flask import Flask, render_template, request, jsonify, url_for
from flask_socketio import SocketIO, join_room, emit
import cv2
import numpy as np
import dlib
from imutils import face_utils
import base64
import time
import subprocess
import os
import re
from textblob import TextBlob
from collections import Counter
from uuid import uuid4
import hmac
import hashlib
import json

app = Flask(__name__)
socketio = SocketIO(app)

# TalkJS configuration
TALKJS_APP_ID = "YOUR_TALKJS_APP_ID"  # Replace with your TalkJS App ID
TALKJS_SECRET_KEY = "null"  # Replace with your TalkJS Secret Key

# Initialize models for face and gaze detection
try:
    if not os.path.exists("models/shape_predictor_68_face_landmarks.dat"):
        raise FileNotFoundError("Shape predictor file not found. Please download it.")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    print("Face detection models loaded successfully")
except Exception as e:
    print(f"Failed to load face detection models: {e}")
    detector = None
    predictor = None

# Store states
code_state = {"content": "", "timestamp": 0}
notes_state = {"content": "", "timestamp": 0}
clients = {}  # {sid: role}
sessions = {}  # {session_id: {'interviewer_sid': sid, 'jobseeker_sid': None, 'created_at': timestamp}}

# Anti-cheating state tracking
cheating_events = {
    "jobseeker": {
        "tab_switches": 0,
        "inactivity_periods": 0,
        "pastes": 0,
        "screen_shares": 0,
        "audio_alerts": 0,
        "last_tab_switch": 0,
        "last_audio_alert": 0,
        "last_inactivity_alert": 0,
        "no_face_detections": 0,
        "gaze_off_screen": 0,
        "multiple_faces": 0,
        "last_no_face_alert": 0,
        "last_gaze_alert": 0,
        "last_multiple_faces_alert": 0
    }
}

# Supported languages
LANGUAGE_CONFIG = {
    "python": {"command": ["python"], "extension": ".py", "mode": "ace/mode/python"},
    "javascript": {"command": ["node"], "extension": ".js", "mode": "ace/mode/javascript"},
    "java": {"command": ["java"], "extension": ".java", "mode": "ace/mode/java", "compile": ["javac"]}
}

# Cooldown for messages
last_no_face_message = {"interviewer": 0, "jobseeker": 0}
MESSAGE_COOLDOWN = 2

def is_ai_generated_code(code):
    lines = code.split('\n')
    code_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    comment_lines = [line.strip() for line in lines if line.strip().startswith('#')]
    
    score = 0
    details = []

    comment_ratio = len(comment_lines) / (len(code_lines) + 1e-5)
    if comment_ratio > 0.5:
        score += 30
        details.append("High comment-to-code ratio (AI-like)")
    elif comment_ratio < 0.1 and len(code_lines) > 5:
        score += 10
        details.append("Low comment density")

    if comment_lines:
        comment_text = ' '.join(comment_lines)
        blob = TextBlob(comment_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.3:
            score += 20
            details.append("Overly positive comments (AI-like)")

    tokens = re.findall(r'\b\w+\b', code)
    token_freq = Counter(tokens)
    common_generic_vars = {'x', 'y', 'temp', 'result', 'data', 'i', 'j'}
    generic_var_count = sum(token_freq[var] for var in common_generic_vars)
    if generic_var_count / (len(tokens) + 1e-5) > 0.3:
        score += 20
        details.append("High use of generic variable names (AI-like)")

    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if indent_levels and max(indent_levels) - min(indent_levels) < 2:
        score += 20
        details.append("Highly uniform indentation (AI-like)")

    import_count = len([line for line in code_lines if line.startswith(('import ', 'from '))])
    if import_count > 5 and import_count / len(code_lines) > 0.2:
        score += 15
        details.append("Excessive imports relative to code length (AI-like)")

    try_except_count = code.count('try:') + code.count('except')
    if try_except_count > 3 or (try_except_count > 0 and len(code_lines) < 20):
        score += 15
        details.append("Overuse of try-except blocks (AI-like)")

    line_patterns = Counter([line.strip() for line in code_lines if line.strip()])
    if any(count > 3 for count in line_patterns.values()) and len(code_lines) > 10:
        score += 10
        details.append("Repetitive code patterns (AI-like)")

    long_lines = sum(1 for line in code_lines if len(line) > 80 and '=' in line)
    if long_lines / (len(code_lines) + 1e-5) > 0.2:
        score += 15
        details.append("High proportion of complex one-liners (AI-like)")

    quirky_keywords = {'print(', 'TODO', 'FIXME', '# debug', 'assert'}
    quirk_count = sum(code.count(kw) for kw in quirky_keywords)
    if quirk_count == 0 and len(code_lines) > 15:
        score += 10
        details.append("Lack of personal coding quirks (AI-like)")

    confidence = min(score, 100)
    is_ai_likely = confidence > 50

    return (is_ai_likely, confidence, details)

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"Client connected: {sid}")
    emit('code_update', code_state["content"])
    emit('notes_update', notes_state["content"])

@socketio.on('join_role')
def handle_join_role(data):
    sid = request.sid
    role = data.get('role', 'jobseeker')
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        emit('error', {'message': 'Invalid or expired session ID'}, to=sid)
        return
    
    clients[sid] = role
    join_room(session_id)
    if role == 'interviewer':
        sessions[session_id]['interviewer_sid'] = sid
        emit('cheating_stats_update', cheating_events["jobseeker"], to=sid)
        emit('join_link', {'link': url_for('join_interview', session_id=session_id, _external=True)}, to=sid)
    else:
        sessions[session_id]['jobseeker_sid'] = sid
    
    print(f"Client {sid} joined as {role} in session {session_id}")
    emit('role_assigned', {'role': role, 'session_id': session_id}, to=sid)

@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    role = data.get('role')
    frame = data.get('frame')
    session_id = data.get('session_id')
    
    if not session_id or session_id not in sessions:
        socketio.emit('error', {'message': 'Invalid session ID'}, to=sid)
        return
    
    if not frame or not isinstance(frame, str):
        print(f"Received invalid frame from {role} ({sid})")
        socketio.emit('interviewer_message', f"Invalid frame received from {role}", room=session_id)
        return
    
    print(f"Received {role} frame size: {len(frame)} bytes")
    process_video_frame(frame, is_interviewer=(role == 'interviewer'), session_id=session_id)
    
    target_role = 'jobseeker' if role == 'interviewer' else 'interviewer'
    target_sid = sessions[session_id]['jobseeker_sid'] if role == 'interviewer' else sessions[session_id]['interviewer_sid']
    if target_sid and target_sid in clients:
        emit('video_frame', {'role': role, 'frame': frame}, to=target_sid)
        print(f"Broadcasted {role} frame to {target_role}")

@socketio.on('mute_audio')
def handle_mute_audio(data):
    role = data.get('role')
    muted = data.get('muted')
    print(f"{role} audio muted: {muted}")
    emit('audio_status', {'role': role, 'muted': muted}, broadcast=True)

@socketio.on('toggle_video')
def handle_toggle_video(data):
    role = data.get('role')
    enabled = data.get('enabled')
    print(f"{role} video enabled: {enabled}")
    emit('video_status', {'role': role, 'enabled': enabled}, broadcast=True)

def process_video_frame(data, is_interviewer, session_id):
    if detector is None or predictor is None:
        socketio.emit('interviewer_message', "Gaze detection unavailable", room=session_id)
        return

    try:
        if not isinstance(data, str) or ',' not in data:
            raise ValueError("Invalid frame data format")
        
        parts = data.split(',')
        frame_data = base64.b64decode(parts[1])
        np_frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        current_time = time.time()
        role_key = "interviewer" if is_interviewer else "jobseeker"

        if not is_interviewer:
            if len(rects) > 1:
                if current_time - cheating_events["jobseeker"]["last_multiple_faces_alert"] >= MESSAGE_COOLDOWN:
                    cheating_events["jobseeker"]["multiple_faces"] += 1
                    cheating_events["jobseeker"]["last_multiple_faces_alert"] = current_time
                    socketio.emit('interviewer_message', "Multiple Faces Detected!", room=session_id)
                    socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            elif len(rects) == 0:
                if current_time - cheating_events["jobseeker"]["last_no_face_alert"] >= MESSAGE_COOLDOWN:
                    cheating_events["jobseeker"]["no_face_detections"] += 1
                    cheating_events["jobseeker"]["last_no_face_alert"] = current_time
                    socketio.emit('interviewer_message', "No Face Detected!", room=session_id)
                    socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                left_eye_center = left_eye.mean(axis=0).astype(int)
                right_eye_center = right_eye.mean(axis=0).astype(int)
                frame_width = frame.shape[1]
                gaze_threshold = frame_width // 4

                if (left_eye_center[0] < gaze_threshold or left_eye_center[0] > frame_width - gaze_threshold or
                    right_eye_center[0] < gaze_threshold or right_eye_center[0] > frame_width - gaze_threshold):
                    if current_time - cheating_events["jobseeker"]["last_gaze_alert"] >= MESSAGE_COOLDOWN:
                        cheating_events["jobseeker"]["gaze_off_screen"] += 1
                        cheating_events["jobseeker"]["last_gaze_alert"] = current_time
                        socketio.emit('interviewer_message', "Gaze Off-Screen!", room=session_id)
                        socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)

    except Exception as e:
        socketio.emit('interviewer_message', f"Error processing frame: {str(e)}", room=session_id)

@socketio.on('code_change')
def handle_code_change(data):
    global code_state
    if isinstance(data, dict) and "content" in data and "timestamp" in data:
        if data["timestamp"] > code_state["timestamp"]:
            code_state = {"content": data["content"], "timestamp": data["timestamp"]}
            socketio.emit('code_update', code_state["content"])

@socketio.on('notes_change')
def handle_notes_change(data):
    global notes_state
    if isinstance(data, dict) and "content" in data and "timestamp" in data:
        if data["timestamp"] > notes_state["timestamp"]:
            notes_state = {"content": data["content"], "timestamp": data["timestamp"]}
            socketio.emit('notes_update', notes_state["content"])

@socketio.on('interviewer_message')
def handle_interviewer_message(message):
    current_time = time.time()
    for session_id, session in sessions.items():
        if session['interviewer_sid'] and session['interviewer_sid'] in clients:
            if "Job Seeker switched tabs" in message:
                if current_time - cheating_events["jobseeker"]["last_tab_switch"] > 5:
                    cheating_events["jobseeker"]["tab_switches"] += 1
                    cheating_events["jobseeker"]["last_tab_switch"] = current_time
                    socketio.emit('interviewer_message', message, room=session_id)
                    socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            elif "Job Seeker inactive" in message:
                if current_time - cheating_events["jobseeker"]["last_inactivity_alert"] > 10:
                    cheating_events["jobseeker"]["inactivity_periods"] += 1
                    cheating_events["jobseeker"]["last_inactivity_alert"] = current_time
                    socketio.emit('interviewer_message', message, room=session_id)
                    socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            elif "Job Seeker pasted" in message:
                cheating_events["jobseeker"]["pastes"] += 1
                socketio.emit('interviewer_message', message, room=session_id)
                socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            elif "window size suggests possible screen sharing" in message:
                cheating_events["jobseeker"]["screen_shares"] += 1
                socketio.emit('interviewer_message', message, room=session_id)
                socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            elif "Unexpected audio activity" in message:
                if current_time - cheating_events["jobseeker"]["last_audio_alert"] > 10:
                    cheating_events["jobseeker"]["audio_alerts"] += 1
                    cheating_events["jobseeker"]["last_audio_alert"] = current_time
                    socketio.emit('interviewer_message', message, room=session_id)
                    socketio.emit('cheating_stats_update', cheating_events["jobseeker"], room=session_id)
            else:
                socketio.emit('interviewer_message', message, room=session_id)
            break

@app.route('/compile', methods=['POST'])
def compile_code():
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', 'python').lower()

        if language not in LANGUAGE_CONFIG:
            return jsonify({"error": f"Unsupported language: {language}"}), 400
        if not code.strip():
            return jsonify({"error": "No code provided"}), 400

        is_ai, confidence, ai_details = is_ai_generated_code(code)
        ai_result = {
            "is_ai_generated": is_ai,
            "confidence": confidence,
            "details": ai_details
        }

        config = LANGUAGE_CONFIG[language]
        temp_file = os.path.join(os.path.dirname(__file__), f"temp_code_{int(time.time() * 1000)}{config['extension']}")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)

        if "compile" in config:
            compile_command = config["compile"] + [temp_file]
            compile_result = subprocess.run(
                compile_command,
                capture_output=True,
                text=True,
                timeout=5
            )
            if compile_result.stderr:
                os.remove(temp_file)
                return jsonify({"error": compile_result.stderr, "ai_result": ai_result}), 400

        run_command = config["command"]
        if language == "java":
            class_name = temp_file.replace('.java', '')
            run_command.append(class_name)
        else:
            run_command.append(temp_file)

        if language == "python":
            try:
                subprocess.run(["python", "--version"], capture_output=True, text=True, timeout=5)
            except FileNotFoundError:
                run_command[0] = "python3"

        result = subprocess.run(
            run_command,
            capture_output=True,
            text=True,
            timeout=5
        )

        output = result.stdout if result.stdout else ""
        error = result.stderr if result.stderr else ""

        try:
            os.remove(temp_file)
            if language == "java" and os.path.exists(class_name + ".class"):
                os.remove(class_name + ".class")
        except Exception as e:
            print(f"Failed to delete temp files: {e}")

        if error:
            return jsonify({"error": error, "ai_result": ai_result})
        return jsonify({"output": output, "ai_result": ai_result})

    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_file)
        except:
            pass
        return jsonify({"error": "Execution timed out", "ai_result": ai_result}), 500
    except Exception as e:
        try:
            os.remove(temp_file)
        except:
            pass
        return jsonify({"error": f"Execution failed: {str(e)}", "ai_result": ai_result}), 500

@app.route('/talkjs_user', methods=['GET'])
def get_talkjs_user():
    try:
        sid = request.sid if hasattr(request, 'sid') else None
        if not sid or sid not in clients:
            return jsonify({"error": "User not authenticated"}), 401
        
        role = clients[sid]
        user_id = f"{sid}_{role}"
        user_name = role.capitalize()
        
        user_data = {
            "id": user_id,
            "name": user_name,
            "email": f"{user_id}@interview.rookierise.com",
            "role": role
        }
        
        message = json.dumps(user_data, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            TALKJS_SECRET_KEY.encode('utf-8'),
            message,
            hashlib.sha256
        ).hexdigest()
        
        return jsonify({
            "user": user_data,
            "signature": signature,
            "appId": TALKJS_APP_ID
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate TalkJS user: {str(e)}"}), 500

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in clients:
        role = clients[sid]
        for session_id, session in sessions.items():
            if session['interviewer_sid'] == sid or session['jobseeker_sid'] == sid:
                if role == 'interviewer':
                    session['interviewer_sid'] = None
                else:
                    session['jobseeker_sid'] = None
                print(f"{role} disconnected from session {session_id}")
                if not session['interviewer_sid'] and not session['jobseeker_sid']:
                    del sessions[session_id]
                    print(f"Session {session_id} closed")
                break
        del clients[sid]
        print(f"Client {sid} ({role}) disconnected")
        emit('video_status', {'role': role, 'enabled': False}, broadcast=True)
        emit('audio_status', {'role': role, 'muted': True}, broadcast=True)

@app.route('/')
def index():
    try:
        session_id = str(uuid4())
        sessions[session_id] = {
            'interviewer_sid': None,
            'jobseeker_sid': None,
            'created_at': time.time()
        }
        return render_template('index.html', session_id=session_id, role='interviewer')
    except Exception as e:
        return jsonify({'error': f'Failed to create session: {str(e)}'}), 500

@app.route('/create_interview', methods=['POST'])
def create_interview():
    try:
        session_id = str(uuid4())
        sessions[session_id] = {
            'interviewer_sid': None,
            'jobseeker_sid': None,
            'created_at': time.time()
        }
        join_link = url_for('join_interview', session_id=session_id, _external=True)
        return jsonify({'session_id': session_id, 'join_link': join_link}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to create interview: {str(e)}'}), 500

@app.route('/join/<session_id>')
def join_interview(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session ID'}), 404
    return render_template('index.html', session_id=session_id, role='jobseeker')

@app.route('/cheating_stats', methods=['GET'])
def get_cheating_stats():
    return jsonify(cheating_events["jobseeker"])

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)