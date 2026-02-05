import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import json
import os
import math
from datetime import datetime
from threading import Thread

class WebcamVideoStream:
    """Threaded video stream for non-blocking frame capture."""
    def __init__(self, src=0, backend=None):
        if backend is not None:
            self.stream = cv2.VideoCapture(src, backend)
        else:
            self.stream = cv2.VideoCapture(src)
        
        # Disable buffering for lowest latency
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

    def set(self, prop, value):
        return self.stream.set(prop, value)

class KalmanHandTracker:
    """Kalman Filter for hand position/velocity prediction."""
    def __init__(self):
        # State: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.initialized = False
        self.last_prediction = None

    def update(self, x, y):
        """Update with actual measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        self.last_prediction = (x, y)

    def predict(self):
        """Predict next position based on velocity."""
        if not self.initialized:
            return None
        prediction = self.kf.predict()
        x, y = float(prediction[0][0]), float(prediction[1][0])  # Extract from 2D array
        self.last_prediction = (x, y)
        return (x, y)

    def get_last(self):
        """Get last known position."""
        return self.last_prediction


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_elapsed = t - self.t_prev
        if t_elapsed <= 0: return self.x_prev
        
        a_d = self.alpha(t_elapsed, self.d_cutoff)
        dx = (x - self.x_prev) / t_elapsed
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(t_elapsed, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def alpha(self, te, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

class BlobTracker:
    """Fallback blob tracking using motion detection."""
    def __init__(self):
        self.prev_frame = None
        
    def find_hand_blob(self, frame, last_x, last_y, search_radius=200):
        """Find largest moving blob near last known position."""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame differencing
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.prev_frame = gray
            return None
            
        # Filter contours near last known position
        height, width = frame.shape[:2]
        last_px = int(last_x * width)
        last_py = int(last_y * height)
        
        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # Minimum area
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Check if near last position
            dist = np.sqrt((cx - last_px)**2 + (cy - last_py)**2)
            if dist < search_radius:
                valid_contours.append((cnt, cx, cy, cv2.contourArea(cnt)))
        
        if not valid_contours:
            self.prev_frame = gray
            return None
            
        # Get largest valid contour
        largest = max(valid_contours, key=lambda x: x[3])
        _, cx, cy, _ = largest
        
        self.prev_frame = gray
        return (cx / width, cy / height)

class HandState:
    """Stores the state for a single hand (Left or Right)."""
    def __init__(self):
        self.count = 0
        self.stage = "Neutral"
        self.filter = None  # OneEuroFilter initialized on first frame
        
        # Fail-safe tracking
        self.tracking_mode = "CONFIRMED"  # CONFIRMED, PREDICTED, BLOB_FALLBACK, LERP
        self.last_velocity = (0.0, 0.0)
        self.prediction_frames = 0
        self.max_prediction_frames = 10
        self.lerp_progress = 0.0
        self.lerp_start_pos = None
        self.lerp_target_pos = None

    def smooth_value(self, t, val):
        if self.filter is None:
            self.filter = OneEuroFilter(t, val, min_cutoff=0.1, beta=0.1)
            return val
        return self.filter(t, val)

class HandRepCounter:
    def __init__(self):
        # MediaPipe Setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.4, # Lower for fast recovery
            min_tracking_confidence=0.4,
            model_complexity=0, 
            max_num_hands=2
        )

        # State management for both hands
        self.hand_states = {
            "Left": HandState(),
            "Right": HandState()
        }

        # Threshold (Percentage of screen height)
        self.threshold_pct = 0.50 

        # Game State Variables
        self.state = "TITLE" # TITLE, CAMERA_SELECT, COUNTDOWN, PLAYING, NAME_ENTRY, GAME_OVER
        self.start_time = 0
        self.game_duration = 15 # seconds
        self.countdown_start = 0
        self.countdown_duration = 3
        self.input_name = ""  # Captures typed name
        self.selected_camera = None  # Will be set in camera select screen

        # Performance Settings (MODULAR)
        self.camera_exposure = -7 
        self.camera_gain = 100
        self.inference_width = 480 
        self.inference_height = 360
        self.fps_log = deque(maxlen=30)

        # Predictive Tracking Settings
        self.process_every_n_frames = 2  # TUNABLE: Process AI every N frames
        self.frame_counter = 0
        self.kalman_trackers = {
            "Left": KalmanHandTracker(),
            "Right": KalmanHandTracker()
        }
        self.blob_trackers = {
            "Left": BlobTracker(),
            "Right": BlobTracker()
        }
        self.roi_boxes = {}  # Store ROI for each hand
        self.roi_size = 400
        
        # Fail-Safe Tracking
        self.hand_positions = {}  # Store last known positions for each hand

        # Tracking Recovery
        self.last_results = None
        self.lost_frames = 0
        self.max_lost_frames = 3

        # Scoreboard
        self.scoreboard_file = "scoreboard.json"
        self.scores = self.load_scores()

        # Custom Splash Image
        self.splash_img_path = "/Users/brandontautuan/.gemini/antigravity/brain/f1a6865f-dcf4-44a6-acc9-a9ef0550f829/uploaded_media_1770240472819.jpg"
        self.splash_img = cv2.imread(self.splash_img_path)
        if self.splash_img is not None:
            self.splash_img = cv2.resize(self.splash_img, (1280, 720))

    def find_camo_camera(self):
        """Automatically discover Camo virtual camera by testing indices 0-5."""
        print("Searching for Camo camera...")
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Try to get camera name/backend info
                    backend = cap.getBackendName()
                    print(f"  Camera {idx}: {backend} - Testing...")
                    
                    # Camo cameras typically support high resolutions
                    # Test if we can set high resolution as indicator
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    
                    if actual_width >= 1280:  # Likely a good camera
                        print(f"  ✓ Found high-quality camera at index {idx}")
                        cap.release()
                        return idx
                cap.release()
        
        print("  ! No Camo camera found, defaulting to index 0")
        return 0

    def load_scores(self):
        """Loads scores from JSON file."""
        if os.path.exists(self.scoreboard_file):
            try:
                with open(self.scoreboard_file, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_score(self, final_score, name):
        """Saves current score to the leaderboard."""
        if not name.strip():
            name = "Player"
            
        new_entry = {
            "name": name,
            "score": final_score,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.scores.append(new_entry)
        # Sort by score descending
        self.scores.sort(key=lambda x: x["score"], reverse=True)
        # Keep Top 10
        self.scores = self.scores[:10]
        
        with open(self.scoreboard_file, "w") as f:
            json.dump(self.scores, f, indent=4)

    def lerp(self, start, end, alpha):
        """Linear interpolation between two values."""
        return start + (end - start) * alpha
    
    def get_tracking_color(self, mode):
        """Get color based on tracking mode for visual debugging."""
        colors = {
            "CONFIRMED": (0, 255, 0),      # Green
            "PREDICTED": (0, 0, 255),      # Red
            "BLOB_FALLBACK": (0, 165, 255), # Orange
            "LERP": (0, 255, 255)          # Yellow
        }
        return colors.get(mode, (255, 255, 255))

    def process_hand_with_failsafe(self, label, landmark_x, landmark_y, curr_time, width, height, display_image):
        """Process a single hand with fail-safe tracking modes."""
        current_hand = self.hand_states[label]
        kalman = self.kalman_trackers[label]
        
        # Check if transitioning from prediction back to confirmation
        if current_hand.tracking_mode in ["PREDICTED", "BLOB_FALLBACK"]:
            current_hand.tracking_mode = "LERP"
            current_hand.lerp_start_pos = self.hand_positions.get(label, (landmark_x, landmark_y))
            current_hand.lerp_target_pos = (landmark_x, landmark_y)
            current_hand.lerp_progress = 0.0
        
        # Handle LERP transition
        if current_hand.tracking_mode == "LERP":
            current_hand.lerp_progress += 0.2  # 5 frames to complete
            if current_hand.lerp_progress >= 1.0:
                current_hand.tracking_mode = "CONFIRMED"
                final_x, final_y = landmark_x, landmark_y
            else:
                final_x = self.lerp(current_hand.lerp_start_pos[0], landmark_x, current_hand.lerp_progress)
                final_y = self.lerp(current_hand.lerp_start_pos[1], landmark_y, current_hand.lerp_progress)
        else:
            current_hand.tracking_mode = "CONFIRMED"
            final_x, final_y = landmark_x, landmark_y
        
        # Update velocity
        if label in self.hand_positions:
            prev_x, prev_y = self.hand_positions[label]
            current_hand.last_velocity = (final_x - prev_x, final_y - prev_y)
        
        self.hand_positions[label] = (final_x, final_y)
        current_hand.prediction_frames = 0
        kalman.update(final_x, final_y)
        
        # Game logic
        smooth_y = current_hand.smooth_value(curr_time, final_y)
        if smooth_y < self.threshold_pct: current_hand.stage = "Up"
        elif smooth_y > self.threshold_pct:
            if current_hand.stage == "Up":
                current_hand.stage = "Down"
                current_hand.count += 1
        
        # Draw marker with color coding
        cx, cy = int(final_x * width), int(final_y * height)
        color = self.get_tracking_color(current_hand.tracking_mode)
        cv2.circle(display_image, (cx, cy), 15, color, -1)
        cv2.putText(display_image, current_hand.tracking_mode[:4], (cx-20, cy-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return cx, cy

    def handle_lost_hand(self, label, curr_time, width, height, display_image, image):
        """Handle a hand that wasn't detected - activate fail-safe."""
        current_hand = self.hand_states[label]
        current_hand.prediction_frames += 1
        
        if current_hand.prediction_frames > current_hand.max_prediction_frames:
            return None  # Too many frames lost
        
        if label not in self.hand_positions:
            return None
        
        last_x, last_y = self.hand_positions[label]
        vx, vy = current_hand.last_velocity
        
        # Velocity-based prediction
        pred_x = max(0.0, min(1.0, last_x + vx))
        pred_y = max(0.0, min(1.0, last_y + vy))
        current_hand.tracking_mode = "PREDICTED"
        
        # Try blob tracking fallback
        blob_tracker = self.blob_trackers[label]
        blob_result = blob_tracker.find_hand_blob(image, last_x, last_y)
        
        if blob_result:
            pred_x = 0.7 * pred_x + 0.3 * blob_result[0]
            pred_y = 0.7 * pred_y + 0.3 * blob_result[1]
            current_hand.tracking_mode = "BLOB_FALLBACK"
        
        self.hand_positions[label] = (pred_x, pred_y)
        
        # Game logic with predicted position
        smooth_y = current_hand.smooth_value(curr_time, pred_y)
        if smooth_y < self.threshold_pct: current_hand.stage = "Up"
        elif smooth_y > self.threshold_pct:
            if current_hand.stage == "Up":
                current_hand.stage = "Down"
                current_hand.count += 1
        
        # Draw predicted marker
        cx, cy = int(pred_x * width), int(pred_y * height)
        color = self.get_tracking_color(current_hand.tracking_mode)
        cv2.circle(display_image, (cx, cy), 15, color, -1)
        cv2.putText(display_image, current_hand.tracking_mode[:4], (cx-20, cy-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return (cx, cy)

    def reset_game(self):
        """Resets counters and states for a new game."""
        self.hand_states["Left"] = HandState()
        self.hand_states["Right"] = HandState()
        self.state = "COUNTDOWN"
        self.countdown_start = time.time()

    def draw_visuals(self, image, height, width, time_left):
        """Draws game HUD during PLAYING state."""
        
        thresh_y = int(height * self.threshold_pct)

        # Draw Threshold Line
        cv2.line(image, (0, thresh_y), (width, thresh_y), (255, 0, 0), 2)
        cv2.putText(image, 'THRESHOLD', (int(width/2) - 60, thresh_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Calculate Total Paired Score
        left_state = self.hand_states["Left"]
        right_state = self.hand_states["Right"]
        total_score = min(left_state.count, right_state.count)

        # Draw Counters (Left/Right)
        # Left
        cv2.rectangle(image, (0, 0), (160, 80), (50, 50, 50), -1)
        cv2.putText(image, 'LEFT', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        l_stage_color = (0, 255, 0) if left_state.stage == "Up" else (0, 0, 255)
        cv2.putText(image, str(left_state.count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, left_state.stage, (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_stage_color, 2, cv2.LINE_AA)

        # Right
        cv2.rectangle(image, (width - 160, 0), (width, 80), (50, 50, 50), -1)
        cv2.putText(image, 'RIGHT', (width - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        r_stage_color = (0, 255, 0) if right_state.stage == "Up" else (0, 0, 255)
        text_w_count = cv2.getTextSize(str(right_state.count), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0]
        cv2.putText(image, str(right_state.count), (width - 20 - text_w_count, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, right_state.stage, (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, r_stage_color, 2, cv2.LINE_AA)

        # Draw Center Info (Score + Timer)
        cv2.rectangle(image, (int(width/2) - 80, 0), (int(width/2) + 80, 90), (0, 0, 0), -1)
        
        # Score
        cv2.putText(image, 'SCORE', (int(width/2) - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        score_text = str(total_score)
        text_w_score = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0][0]
        cv2.putText(image, score_text, (int(width/2) - int(text_w_score/2), 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # Timer
        timer_text = f"{time_left:.1f}s"
        text_w_timer = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
        color_timer = (0, 255, 0) if time_left > 5 else (0, 0, 255)
        cv2.putText(image, timer_text, (int(width/2) - int(text_w_timer/2), 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_timer, 2, cv2.LINE_AA)


    def draw_text_centered(self, image, text, font, scale, color, thickness, y_pos_pct):
        """Helper to draw centered text on any image."""
        height, width, _ = image.shape
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = int(height * y_pos_pct)
        cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

    def run(self):
        window_name = 'How fast can you do 6-7'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Wait for camera selection (will be set in CAMERA_SELECT state)
        # Start with a dummy camera just for the UI
        vs = None
        
        prev_time = time.time()

        while True:
            # Calculate FPS
            curr_time = time.time()
            elapsed_frame = curr_time - prev_time
            prev_time = curr_time
            if elapsed_frame > 0:
                self.fps_log.append(1.0 / elapsed_frame)
            avg_fps = sum(self.fps_log) / len(self.fps_log) if self.fps_log else 0

            # Handle Inputs (Non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Check if window was closed via 'X' button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Initialize camera after selection
            if self.selected_camera is not None and vs is None:
                print(f"Initializing camera {self.selected_camera}...")
                vs = WebcamVideoStream(src=self.selected_camera, backend=cv2.CAP_AVFOUNDATION)
                
                # Force 60 FPS for high-speed tracking
                vs.set(cv2.CAP_PROP_FPS, 60)
                
                # Set high resolution
                vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Disable auto settings to reduce motion blur and lag
                vs.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                vs.set(cv2.CAP_PROP_EXPOSURE, self.camera_exposure)
                vs.set(cv2.CAP_PROP_GAIN, self.camera_gain)
                vs.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                
                vs.start()
                time.sleep(0.5)  # Warm up camera

                # Get actual FPS
                actual_fps = vs.stream.get(cv2.CAP_PROP_FPS)
                print(f"Camera Ready. Index: {self.selected_camera}, FPS: {actual_fps}, Exposure: {self.camera_exposure}")

            # --- TITLE SCREEN ---
            if self.state == "TITLE":
                # Create background (either splash or black)
                if self.splash_img is not None:
                    # Dynamically resize splash to match current capture resolution (or vice versa)
                    # For title, let's use 1280x720
                    frame = cv2.resize(self.splash_img, (1280, 720))
                else:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                # Semi-transparent overlay for text readability
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 300), (1280, 500), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                self.draw_text_centered(frame, "Can you do 6-7 the fastest?", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, 0.45)
                self.draw_text_centered(frame, "Press ENTER to Play", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, 0.6)
                
                cv2.imshow(window_name, frame)
                
                if key == 13: # Enter Key
                    self.state = "CAMERA_SELECT"

            # --- CAMERA SELECT SCREEN ---
            elif self.state == "CAMERA_SELECT":
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                self.draw_text_centered(frame, "SELECT CAMERA", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, 0.3)
                self.draw_text_centered(frame, "Press 1 for Mac Camera", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, 0.5)
                self.draw_text_centered(frame, "Press 2 for Phone Camera (Camo)", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, 0.6)
                
                cv2.imshow(window_name, frame)
                
                if key == ord('1'):
                    self.selected_camera = 0
                    print("Selected: Mac Camera (index 0)")
                    self.reset_game()
                elif key == ord('2'):
                    # Find Camo camera
                    self.selected_camera = self.find_camo_camera()
                    print(f"Selected: Phone Camera (index {self.selected_camera})")
                    self.reset_game()

            # --- NAME ENTRY SCREEN ---
            elif self.state == "NAME_ENTRY":
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                final_score = min(self.hand_states["Left"].count, self.hand_states["Right"].count)
                
                self.draw_text_centered(frame, "NEW SCORE!", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, 0.2)
                self.draw_text_centered(frame, f"{final_score} REPS", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5, 0.4)
                
                self.draw_text_centered(frame, "Enter Your Name:", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, 0.6)
                
                # Display typing string
                display_name = self.input_name + "_"
                self.draw_text_centered(frame, display_name, 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4, 0.75)
                
                self.draw_text_centered(frame, "Press ENTER to Save", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1, 0.9)

                cv2.imshow(window_name, frame)
                
                # Input Handling for Name Entry
                if key == 13: # Enter
                    self.save_score(final_score, self.input_name)
                    self.state = "GAME_OVER"
                elif key == 8 or key == 127: # Backspace (Mac/Linux)
                    self.input_name = self.input_name[:-1]
                elif 32 <= key <= 126: # Regular Characters
                    if len(self.input_name) < 15:
                        self.input_name += chr(key)

            # --- GAME OVER SCREEN ---
            elif self.state == "GAME_OVER":
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                final_score = min(self.hand_states["Left"].count, self.hand_states["Right"].count)
                
                self.draw_text_centered(frame, "JOIN FLC++!", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, 0.15)
                self.draw_text_centered(frame, f"FINAL SCORE: {final_score}", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4, 0.3)

                # Leaderboard Section
                self.draw_text_centered(frame, "LOCAL TOP SCORES", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, 0.45)
                
                for i, entry in enumerate(self.scores[:5]):
                    name = entry.get('name', 'Player')
                    rank_text = f"#{i+1} - {name}: {entry['score']} reps"
                    
                    # Highlight if it's the current player's just-saved score
                    color = (0, 255, 215) if entry['score'] == final_score and name == self.input_name else (255, 255, 255)
                    self.draw_text_centered(frame, rank_text, 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 0.52 + (i * 0.06))

                self.draw_text_centered(frame, "Press ENTER to Play Again", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, 0.9)

                cv2.imshow(window_name, frame)
                
                if key == 13: # Enter Key
                    if vs is not None:
                        vs.stop()
                        vs = None
                    self.selected_camera = None
                    self.state = "TITLE" # Loop back to title

            # --- COUNTDOWN & PLAYING (Camera Active) ---
            else:
                # Read from threaded stream
                frame = vs.read()
                if frame is None: continue

                # Flip & Convert
                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Copy for drawing
                display_image = image.copy()
                height, width, _ = display_image.shape

                # --- TRACKING ---
                if self.state == "PLAYING":
                    self.frame_counter += 1
                    should_process = (self.frame_counter % self.process_every_n_frames == 0)
                    
                    if should_process:
                        # Optimization: Resize for inference
                        inference_frame = cv2.resize(image_rgb, (self.inference_width, self.inference_height))
                        
                        # AI Image Boosting
                        inference_frame = cv2.convertScaleAbs(inference_frame, alpha=1.2, beta=30)
                        
                        results = self.hands.process(inference_frame)
                        
                        # Detection Recovery Logic
                        if results.multi_hand_landmarks:
                            self.last_results = results
                            self.lost_frames = 0
                        else:
                            if self.last_results and self.lost_frames < self.max_lost_frames:
                                results = self.last_results
                                self.lost_frames += 1
                            else:
                                self.last_results = None
                        
                        
                        # Track which hands were detected
                        detected_hands = set()
                        
                        if results and results.multi_hand_landmarks and results.multi_handedness:
                            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                label = handedness.classification[0].label
                                if label not in self.hand_states: continue

                                detected_hands.add(label)
                                
                                # Draw landmarks
                                self.mp_drawing.draw_landmarks(display_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                                
                                # Find HIGHEST POINT of hand (minimum Y value across all landmarks)
                                highest_y = min(lm.y for lm in hand_landmarks.landmark)
                                highest_x = next(lm.x for lm in hand_landmarks.landmark if lm.y == highest_y)
                                
                                # Process with fail-safe logic using highest point
                                self.process_hand_with_failsafe(label, highest_x, highest_y, curr_time, width, height, display_image)
                        
                        # Handle hands that weren't detected - FAIL-SAFE ACTIVATION
                        for label in ["Left", "Right"]:
                            if label not in detected_hands:
                                self.handle_lost_hand(label, curr_time, width, height, display_image, image)
                    
                    else:
                        # FRAME SKIPPING: Use Kalman prediction
                        for label in ["Left", "Right"]:
                            kalman = self.kalman_trackers[label]
                            prediction = kalman.predict()
                            
                            if prediction:
                                px, py = prediction
                                current_hand = self.hand_states[label]
                                
                                # Use prediction for game logic
                                smooth_y = current_hand.smooth_value(curr_time, py)
                                
                                if smooth_y < self.threshold_pct: current_hand.stage = "Up"
                                elif smooth_y > self.threshold_pct:
                                    if current_hand.stage == "Up":
                                        current_hand.stage = "Down"
                                        current_hand.count += 1
                                
                                # Draw predicted position (different color)
                                cx, cy = int(px * width), int(py * height)
                                cv2.circle(display_image, (cx, cy), 10, (100, 100, 255), -1)  # Blue for prediction

                # --- HUD & LOGIC ---
                # Show FPS Overlay
                cv2.putText(display_image, f"FPS: {avg_fps:.1f}", (width - 120, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if self.state == "COUNTDOWN":
                    elapsed = time.time() - self.countdown_start
                    count_val = 3 - int(elapsed)
                    
                    self.draw_visuals(display_image, height, width, 15.0) 
                    
                    if count_val > 0:
                        self.draw_text_centered(display_image, str(count_val), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 6.0, (0, 0, 255), 10, 0.55)
                    else:
                        self.draw_text_centered(display_image, "GO!", 
                                               cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 0), 8, 0.55)
                        
                    if elapsed > 3.5:
                        self.state = "PLAYING"
                        self.start_time = time.time()

                elif self.state == "PLAYING":
                    time_left = self.game_duration - (time.time() - self.start_time)
                    if time_left <= 0:
                        time_left = 0
                        self.input_name = ""  # Reset name buffer for entry screen
                        self.state = "NAME_ENTRY"
                    
                    self.draw_visuals(display_image, height, width, time_left)

                cv2.imshow(window_name, display_image)


        self.hands.close()
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    counter = HandRepCounter()
    counter.run()
