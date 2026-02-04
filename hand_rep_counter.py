import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import json
import os
from datetime import datetime

class HandState:
    """Stores the state for a single hand (Left or Right)."""
    def __init__(self, buffer_size=2): # Reduced buffer info
        self.count = 0
        self.stage = "Neutral"
        self.rep_ready = False
        self.y_buffer = deque(maxlen=buffer_size)

    def smooth_value(self, val):
        self.y_buffer.append(val)
        return sum(self.y_buffer) / len(self.y_buffer)

class HandRepCounter:
    def __init__(self):
        # MediaPipe Setup
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, # Lower confidence for speed
            min_tracking_confidence=0.5,
            model_complexity=0, # Use Lite model for speed
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
        self.state = "TITLE" # TITLE, COUNTDOWN, PLAYING, NAME_ENTRY, GAME_OVER
        self.start_time = 0
        self.game_duration = 15 # seconds
        self.countdown_start = 0
        self.countdown_duration = 3
        self.input_name = ""  # Captures typed name

        # Scoreboard
        self.scoreboard_file = "scoreboard.json"
        self.scores = self.load_scores()

        # Custom Splash Image
        self.splash_img_path = "/Users/brandontautuan/.gemini/antigravity/brain/f1a6865f-dcf4-44a6-acc9-a9ef0550f829/uploaded_media_1770240472819.jpg"
        self.splash_img = cv2.imread(self.splash_img_path)
        if self.splash_img is not None:
            self.splash_img = cv2.resize(self.splash_img, (1280, 720))

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

    def reset_game(self):
        """Resets counters and states for a new game."""
        self.hand_states["Left"] = HandState(buffer_size=2)
        self.hand_states["Right"] = HandState(buffer_size=2)
        self.state = "COUNTDOWN"
        self.countdown_start = time.time()

    def draw_visuals(self, image, height, width, time_left):
        """Draws game HUD during PLAYING state."""
        
        thresh_y = int(height * self.threshold_pct)

        # Draw Threshold
        cv2.line(image, (0, thresh_y), (width, thresh_y), (255, 0, 0), 2)
        cv2.putText(image, 'THRESHOLD', (int(width/2) - 40, thresh_y - 10), 
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
        
        cap = cv2.VideoCapture(0)
        # Increase resolution for better full screen quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting Game. Press 'q' to exit.")

        while True:
            # Handle Inputs (Non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Check if window was closed via 'X' button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

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

                self.draw_text_centered(frame, "FLC++ Codes in Python Game 6-7 Lamelo Ball", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, 0.45)
                self.draw_text_centered(frame, "Press ENTER to Play", 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, 0.6)
                
                cv2.imshow(window_name, frame)
                
                if key == 13: # Enter Key
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
                
                self.draw_text_centered(frame, "Thanks For Playing!", 
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
                    self.state = "TITLE" # Loop back to title

            # --- COUNTDOWN & PLAYING (Camera Active) ---
            else:
                ret, frame = cap.read()
                if not ret: continue

                # Flip & Convert
                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Copy for drawing
                display_image = image.copy()
                height, width, _ = display_image.shape

                # --- TRACKING ---
                if self.state == "PLAYING":
                    results = self.hands.process(image_rgb)
                    
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            label = handedness.classification[0].label
                            if label not in self.hand_states: continue

                            current_hand = self.hand_states[label]
                            self.mp_drawing.draw_landmarks(display_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            
                            # Logic
                            landmark_y = hand_landmarks.landmark[8].y
                            smooth_y = current_hand.smooth_value(landmark_y)
                            
                            if smooth_y < self.threshold_pct: current_hand.stage = "Up"
                            elif smooth_y > self.threshold_pct:
                                if current_hand.stage == "Up":
                                    current_hand.stage = "Down"
                                    current_hand.count += 1
                                    
                            cx, cy = int(hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height)
                            cv2.circle(display_image, (cx, cy), 12, (255, 0, 255), -1)

                # --- HUD & LOGIC ---
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    counter = HandRepCounter()
    counter.run()
