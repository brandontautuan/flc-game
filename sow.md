# Task List: Wrist Flick Counter (6-7) Improvements

This document outlines a series of potential enhancements for the "How fast can you do 6-7" game, ranging from core mechanics to visual flair and user experience.

## 🛠 Core Mechanics (Rules & Logic)

- [ ] **Symmetry Bonus**: Implement a "Sync-Sense" mechanic where players get a 1.5x score multiplier if their Left and Right hand counts stay within 2 reps of each other throughout the game.
- [ ] **Velocity-Based Scoring**: Instead of a simple line crossing, award more points for "snappier" flicks. Calculate the peak velocity of landmark 9 during the "Up" phase to determine point value (1-5 points per rep).
- [ ] **Endurance Mode (Survival)**: A new game mode where the timer *adds* 2 seconds for every 10 reps, but the threshold line slowly moves higher, forcing wider movements. Game ends when the timer hits zero or frequency drops below 1 rep/sec.
- [ ] **Strict Form Check**: Use the angle between the wrist, MCP, and fingertips to ensure it's a "flick" and not just the whole arm moving up and down.
- [ ] **Rhythm Mode**: Introduce a pulsing beat. Flicks timed to the "drop" count for double.
- [ ] **Dynamic Threshold**: The line moves up or down based on the player's performance. The faster you go, the higher the line moves, increasing the difficulty.

## 🎨 UI & Visual Aesthetics

- [ ] **Heatmap Trails**: Implement a "comet trail" effect for the tracked dots. The color shifts from Electric Violet (#8B5CF6) to Fire Orange as the flick velocity increases.
- [ ] **Visual Pulse**: The "Glassmorphism" HUD bar should pulse or glow slightly on every successful rep.
- [ ] **Combo Gauge**: Add a curved meter on the left/right of the screen that fills up as "Reps Per Second" (RPS) increases. Breaking the combo triggers a "Power Mode" with screen shake.
- [ ] **Particle Bursts**: Use OpenCV to generate a burst of geometric particles (triangles/squares) at the landmark 9 position every time a count is triggered.
- [ ] **Dynamic Background Lighting**: Instead of a static dark BG, use a subtle radial gradient that brightens or changes hue based on total score milestones (e.g., turns gold at 50 reps).
- [ ] **3D Hand Mesh**: (Optional/Advanced) Use MediaPipe's full landmark set to draw a stylized, low-poly 3D wireframe mesh over the hands for a more "cyberpunk" aesthetic.

## 🕹 Game Features & UX

- [ ] **Gesture-Based Navigation**: 
    - [ ] "High Five" to start the game.
    - [ ] "Thumbs Up" to confirm name entry.
    - [ ] "Peace Sign" to take a screenshot of the leaderboard.
- [ ] **Ghost Mode**: On the PLAYING screen, display a semi-transparent "Ghost Dot" representing the movement of the #1 local leaderboard holder's performance.
- [ ] **Sound Integration**: 
    - [ ] Haptic-like "thump" sound for reps.
    - [ ] Fast-paced synth-wave music that speeds up during the last 5 seconds.
    - [ ] "Perfect!" voice-over for high-speed streaks.
- [ ] **Global Leaderboard**: A Python script to sync `scoreboard.json` with a simple Firebase/Supabase backend to compare scores with other users of the app.
- [ ] **Multiplayer (Split Screen)**: Support for two different people tracking one hand each, competing side-by-side in the same frame.

## 🧪 Calibration & Robustness

- [ ] **Auto-Exposure Lock**: Add a button in Calibration to lock exposure once the user is happy with the lighting, preventing "flicker" during high-speed movements.
- [ ] **Distance Indicator**: Use the distance between landmarks to estimate how far the user is from the camera. Warn them if they are too close/far for optimal tracking.
- [ ] **ROI Visualization**: Draw a subtle box around the detected hand areas to show where the "fast processing" is happening.
