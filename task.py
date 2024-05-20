from typing import NamedTuple, Tuple


class LineBuildingTask(NamedTuple):
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]
    face: str = 'top' # can be 'top', 'bottom', 'front'
    spacing: int = 0 # spacing = number of "voids" between two placed blocks
    y_inc: int = 0
        # (for face = 'front' or 'bottom') y coordinate of the placed blocks
        # relative to the player's feet

class AreaBuildingTask(NamedTuple):
    y: int
    left: int
    top: int
    right: int
    bottom: int