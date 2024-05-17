import pygetwindow
import logger
import time
import win32gui
import win32con
import win32ui
import ctypes
import numpy as np
from typing import Union, List

my_dpi_awareness_context = None

def get_minecraft_window() -> pygetwindow.Win32Window:
    wins = pygetwindow.getWindowsWithTitle('Minecraft')
    if len(wins) == 1:
        w = wins[0]
        logger.info(f'Using window titled "{w.title}" with handle ' + \
            '0x%08X.' % w._hWnd)
        return w
    elif len(wins) > 1:
        prompt = f'Got {len(wins)} windows related to Minecraft. ' + \
            f'Which of the follwing windows is the game window?\n'
        for i, win in enumerate(wins):
            prompt += \
                f'[{i}] {win.title} (handle={"0x%08X" % win._hWnd})\n'
        logger.question(prompt, end='')
        ok = False
        w = None
        while not ok:
            try:
                j = int(input('>>> '))
                w = wins[j]
                ok = True
            except KeyboardInterrupt:
                exit(1)
            except:
                pass
        logger.info(f'Got it! Will use window titled ' + \
            f'"{w.title}" with handle 0x%08X.' % w._hWnd)
        return w
    else:
        logger.error_exit('Error: Cannot find any window related to '
            'Minecraft.')

def adjust_window(w: pygetwindow.Win32Window, x=0, y=0,
    width=1280, height=720, interval=0.002):
    w.activate()
    while not w.isActive:
        time.sleep(interval)
    w.restore()
    time.sleep(2 * interval)
    w.moveTo(x, y)
    while w.left != x or w.top != y:
        time.sleep(interval)
    w.resizeTo(width, height)
    while w.width != width or w.height != height:
        time.sleep(interval)
    
    logger.info(f'Moved window to ({x}, {y}) and '
        f'resized to ({width}, {height}).')

def capture_window(w: pygetwindow.Win32Window, interval=0.01, regions='all') -> \
    Union[np.ndarray, List[np.ndarray]]:
    '''
    w: can be the return value of get_minecraft_window().
    interval: time to sleep between checking when waiting for window to be activated.
    regions:
        - 'all': capture the entire client area.
        - list of (x, y, width, height) tuples: these regions. 
    '''
    ## time_stamps = [[None, time.time_ns()]]
    if not w.isActive:
        w.activate()
    while not w.isActive:
        time.sleep(interval)

    hwnd = w._hWnd
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    ## time_stamps.append(['init', time.time_ns()])

    # adjust DPI awareness
    global my_dpi_awareness_context
    win_dpi_awareness_context = \
        ctypes.windll.user32.GetWindowDpiAwarenessContext(hwnd)
    if my_dpi_awareness_context != win_dpi_awareness_context:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            win_dpi_awareness_context
        )
        my_dpi_awareness_context = win_dpi_awareness_context

    # use BitBlt to capture the window
    client_x, client_y = win32gui.ClientToScreen(hwnd, (0, 0))
    client_rect = win32gui.GetClientRect(hwnd)
    region_list = []
    results = []
    if regions == 'all':
        width = client_rect[2] - client_rect[0]
        height = client_rect[3] - client_rect[1]
        region_list.append(
            ((client_x - w.left, client_y - w.top), (width, height)))
    else:
        for x, y, width, height in regions:
            region_list.append((x, y), (width, height))
    for coordinate, size in region_list:
        bitmap.CreateCompatibleBitmap(dcObj, *size)
        cDC.SelectObject(bitmap)
        ## time_stamps.append(['further init', time.time_ns()])
        cDC.BitBlt((0, 0), size, dcObj, coordinate, win32con.SRCCOPY)
        ## time_stamps.append(['BitBlt', time.time_ns()])
        bits = bitmap.GetBitmapBits(True)
        ## time_stamps.append(['GetBitmapBits', time.time_ns()])
        image = np.frombuffer(bits, dtype=np.uint8) \
            .reshape(height, width, 4)[:, :, np.array([2, 1, 0])]
            # discard alpha channel
        ## time_stamps.append(['frombuffer', time.time_ns()])
        results.append(image)
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

    try: # deleting process should not hinder the function
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(bitmap.GetHandle())
    except:
        pass

    ## time_stamps.append(['final', time.time_ns()])
    ## print(f'Total: {(time_stamps[-1][1] - time_stamps[0][1]) / 1e6:.2f}')
    ## last_time  = time_stamps[0][1]
    ## for i in range(1, len(time_stamps)):
    ##     logger.info(f'{time_stamps[i][0]} used time (ms): ' + \
    ##         f'{(time_stamps[i][1] - last_time) / 1e6:.2f}')
    ##     last_time = time_stamps[i][1]
    if regions == 'all':
        return results[0]
    else:
        return image

w = get_minecraft_window()
adjust_window(w)
time.sleep(1)
w.activate()
before = time.time_ns()
img = capture_window(w)
after = time.time_ns()
print('capture_window used time (ms):', (after - before) / 1e6, 'ms')
import imageio.v3 as iio
iio.imwrite('./text_helper/world_border.png', img)
# for i in range(30):
#     import os
#     os.makedirs('./blue_ice_boat', exist_ok=True)
#     img = capture_window(w)
#     iio.imwrite(f'./blue_ice_boat/{i}.png', img)
#     time.sleep(1)