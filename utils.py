from typing import Iterable
import mouse
import pyautogui
import time


SNEAKING_SPEED = 1.295 # see https://minecraft.fandom.com/wiki/Player#Movement

def extract_args(args: dict, entries: Iterable[str]) -> dict:
    return {k: args[k] for k in entries if k in args}

def mouse_move_rel(dx: int, dy: int):
    mouse._os_mouse.move_relative(dx, dy)

def clip_value(val, bound):
    # clip val into [-bound, bound].
    bound = abs(bound)
    if val > 0:
        return min(val, bound)
    else:
        return max(val, -bound)

def start_moving_forward(sprint=False, interval=0.005):
    if sprint:
        pyautogui.keyDown('w') # press W twice: sprinting
        time.sleep(5 * interval)
        pyautogui.keyUp('w')
        time.sleep(5 * interval)
    pyautogui.keyDown('w')

def stop_moving_forward():
    pyautogui.keyUp('w')

def start_moving_backward():
    pyautogui.keyDown('s')

def stop_moving_backward():
    pyautogui.keyUp('s')

def start_shift():
    pyautogui.keyDown('shift')

def stop_shift():
    pyautogui.keyUp('shift')