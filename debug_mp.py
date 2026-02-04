import mediapipe
print(f"MediaPipe Version: {mediapipe.__version__}")
print(f"Location: {mediapipe.__file__}")
try:
    import mediapipe.python.solutions as solutions
    print("Explicit import of solutions successful.")
except ImportError as e:
    print(f"Explicit import failed: {e}")

try:
    print(f"mp.solutions: {mediapipe.solutions}")
except AttributeError:
    print("mediapipe.solutions not found directly.")
