import pyautogui
import time

time.sleep(10)  # Gives you 3 seconds to open VLC before pressing keys

print("Pressing Play/Pause in VLC...")
pyautogui.press("space")  # Simulate Play/Pause command
print("Done.")
