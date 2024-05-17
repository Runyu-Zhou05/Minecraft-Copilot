from window import capture_window
from pygetwindow import Win32Window

class AutoBuilder:
    '''
    Build paths, platforms, etc. automatically
    in Minecraft.
    '''
    def __init__(self, w: Win32Window):
        self.w = w