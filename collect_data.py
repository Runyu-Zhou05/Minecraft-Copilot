import pyautogui
import time
import random
from window import get_minecraft_window, adjust_window, capture_window
import pydirectinput

def win():
    w = get_minecraft_window()
    adjust_window(w)
    img = capture_window(w)
    return img

for i in range(200): 
    pyautogui.click()
    img = win()
    img.save(f'pictures/{i+392}.png')
    time.sleep(0.5)
