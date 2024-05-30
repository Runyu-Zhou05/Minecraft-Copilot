from window import capture_window
from pygetwindow import Win32Window
from window import adjust_window, capture_window
from text import TextExtractor, match_template, scale_image
from utils import (
    extract_args,
    mouse_move_rel,
    clip_value,
    start_moving_forward,
    stop_moving_forward,
    start_moving_backward,
    stop_moving_backward,
    start_strafing_left,
    stop_strafing_left,
    start_strafing_right,
    stop_strafing_right,
    start_shift,
    stop_shift
)
from typing import Tuple, Union
from task import LineBuildingTask, AreaBuildingTask
import queue
import logger
import traceback
import pynput.keyboard as keyboard
import pyautogui
import time
import math
import numpy as np
import cv2
import threading
import asyncio


class TextPositionInfo(dict):
    '''
    Values should be (x, y, width, height) tuples.
    '''
    MAP = {
        'xyz': 'XYZ',
        'XYZ': 'XYZ',
        'direction': '(Towards',
        'towards': '(Towards',
        '(Towards': '(Towards',
        'tb': 'Targeted Block',
        'targeted_block': 'Targeted Block',
        'Targeted Block': 'Targeted Block',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattribute__(self, name: str):
        if name in TextPositionInfo.MAP:
            return self[TextPositionInfo.MAP[name]]
        elif name == 'gather':
            return lambda keys: [self[k] for k in keys]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value) -> None:
        self[TextPositionInfo.MAP[name]] = value


class AutoBuilder:
    '''
    Build paths, platforms, etc. automatically
    in Minecraft.
    '''
    FIELDS = ['XYZ', '(Towards', 'Targeted Block']

    def __init__(self, w: Win32Window, args: dict={}):
        self.w = w
        self.task_queue = queue.Queue()
        # interrupted: an atomic Boolean variable
        self.interrupted = threading.Event()
        self.reset()
        self.args = args
        self.interrupt_hotkey = args.get('interrupt_hotkey', '<ctrl>+x')
        self.gap = args.get('gap', 50)
            # in milliseconds, like 1000 / fps for capture
        self.interval = args.get('interval', 0.005)
        self.max_mouse_speed = args.get('max_mouse_speed', 10000) # default: unlimited
        self.allow_sprint = args.get('allow_sprint', True)
        self.back = args.get('back', 2) # stand how many blocks behind the targeted block
        '''
        self.xyz = (x, y, z)
        self.direction = (yaw, pitch)
        self.targeted_block = (tx, ty, tz)
        '''
        self.angle_mm_ratio = 0.1508
            # angle: yaw/pitch
            # angle_mm_ratio = delta_angle / mouse_move
            # 0.1508 is obtained on the authors' computer
        self.amr_calibrated = False
            # whether self.angle_mm_ratio has been calibrated

    def reset(self):
        self.interrupted.clear() # whether user pressed interrupt hotkey
        self.xyz = None
        self.direction = None
        self.targeted_block = None
        self.extractor = None
        self.textpos = None
        self.opthread = None
        self.hotkey_thread = None
        self.amr_calibrated = False
        self.require_targeted_block = False
        self.is_idle = True

    def set_interrupted(self):
        logger.error('[purple]AutoBuilder[/purple]: '
            '[yellow]Interrupt signal received.[/yellow]')
        self.interrupted.set()
        self.hotkey_thread.stop()

    def extract_field(self, img: np.ndarray, field: str,
        results: dict, func, width: int):
        '''
        field: e.g. 'XYZ'
        func: e.g. self.extractor.extract_xyz
        width: e.g. 300
        '''
        coord, size = results[field]
        start = coord[0], coord[1] + size[1]
        before = time.time_ns()
        value = func(img, start, (size[0], width * self.extractor.gui_scale))
        after = time.time_ns()
        time_elapsed_ns = after - before
        return value, time_elapsed_ns

    def get_bounding_box(self,
        coord_size: Tuple[Tuple[int, int], Tuple[int, int]],
        width: int):
        coord, size = coord_size
        y, x = coord[0], coord[1] + size[1]
        h, w = size[0], width * self.extractor.gui_scale
        return (x, y, w, h)

    def setup(self):
        '''
        capture the first image and bound the boxes of xyz, directions, etc.
        returns xyz, direction, targeted_block.
        '''
        logger.info('Setting up...', end=' ')
        adjust_window(self.w, show_info=False)
        img = capture_window(self.w)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        extractor_args = ['threshold', 'pixel_value', 'gui_scale']
        self.extractor = TextExtractor(*extract_args(self.args, extractor_args))
        required = ['XYZ', '(Towards', 'Targeted Block']
        if self.extractor.gui_scale is None:
            _, results = self.extractor.guess_gui_scale(img, required=required)
        else:
            _, results = self.extractor.match_templates(img,
                scale=self.extractor.gui_scale, required=required)

        widths = {'XYZ': 300, '(Towards': 220, 'Targeted Block': 160}            

        self.textpos = TextPositionInfo()
            # values: (x, y, width, height) as captured regions
        for field in self.FIELDS:
            if field == '(Towards':
                self.textpos[field] = (56 * self.extractor.gui_scale,
                    results[field][0][0],
                    widths[field] * self.extractor.gui_scale,
                    results[field][1][0])
            else:
                if field in results:
                    self.textpos[field] = self.get_bounding_box(
                        results[field], widths[field])
        
        self.xyz = self.extract_field(img, 'XYZ', results,
            self.extractor.extract_xyz, widths['XYZ'])[0]
        self.direction = self.extract_field(img, '(Towards', results,
            self.extractor.extract_direction, widths['(Towards'])[0]
        if 'Targeted Block' in results:
            self.targeted_block = self.extract_field(img, 'Targeted Block', results,
                self.extractor.extract_targeted_block, widths['Targeted Block'])[0]

        logger.done()

    def try_setup(self, critical=False):
        try:
            self.setup()
        except:
            logger.info('Failed to execute AutoBuilder.setup(), '
                'try to press F3...')
            try:
                self.w.activate()
                while not self.w.isActive:
                    time.sleep(self.interval)
                pyautogui.keyDown('f3')
                pyautogui.keyUp('f3')
                time.sleep(30 * self.interval)
                self.setup()
            except:
                if critical:
                    logger.error('Error occurred while running AutoBuilder.setup():')
                    traceback.print_exc()
                    logger.error_exit('Check if your Minecraft window '
                        'is doing well.')

    def add_task(self, task: Union[LineBuildingTask, AreaBuildingTask]):
        self.task_queue.put(task)

    def turn_to(self, yaw, pitch, epsilon=0.5):
        '''
        yaw: north, south, west, east
            0: z+
            90: x-
            180/-180: z-
            -90: x+
        pitch: up-down
            90: y-
            -90: y+
                                    ^ -x (90 degrees) {west}
                                    |
                                    |
                                    |
        -z (+/-180 degrees) <-------O-------> [z] (0 degrees) {south}
        {north}                     |
                                    |
                                    |
                                   \./
                                   [x] (-90 degrees) {east}
        '''
        logger.info(f'Turning to ({yaw:.1f}/{pitch:.1f})...')
        self.require_targeted_block = False
        while min(abs(self.direction[0] - yaw),
            abs(abs(self.direction[0] - yaw) - 360)) > epsilon or \
            abs(self.direction[1] - pitch) > epsilon:
            cur_yaw, cur_pitch = self.direction
            if abs(yaw - cur_yaw) < 180:
                mm_x = clip_value((yaw - cur_yaw) / self.angle_mm_ratio,
                    self.max_mouse_speed)
            elif abs(yaw - cur_yaw - 360) < 180:
                mm_x = clip_value((yaw - cur_yaw - 360) / self.angle_mm_ratio,
                    self.max_mouse_speed)
            else:
                mm_x = clip_value((yaw - cur_yaw + 360) / self.angle_mm_ratio,
                    self.max_mouse_speed)
            mm_y = clip_value((pitch - cur_pitch) / self.angle_mm_ratio,
                self.max_mouse_speed)
            mouse_move_rel(mm_x, mm_y)
            yield
            new_yaw, new_pitch = self.direction
            delta_yaw = new_yaw - cur_yaw
            delta_pitch = new_pitch - cur_pitch
            if delta_yaw != 0. and delta_pitch != 0.:
                amr, cnt = 0., 0
                if mm_x != 0:
                    cnt += 1
                    amr += delta_yaw / mm_x
                if mm_y != 0:
                    cnt += 1
                    amr += delta_pitch / mm_y
                if cnt > 0 and amr > 0.:
                    amr /= cnt
                    if not self.amr_calibrated:
                        self.angle_mm_ratio = amr
                        self.amr_calibrated = True
                    else:
                        self.angle_mm_ratio = self.angle_mm_ratio * 0.95 + amr * 0.05
        logger.info('Turning done.')

    @staticmethod
    def get_yaw(x_target, z_target, x_cur, z_cur) -> Union[float, None]:
        dx = x_target - x_cur
        dz = z_target - z_cur
        if math.hypot(dx, dz) > 1e-3:
            yaw = round(math.atan2(-dx, dz) / math.pi * 180 * 10) / 10
                # precision: xxx.x
        else:
            yaw = None
        return yaw

    @staticmethod
    def translate_block(x: int, y: int, z: int, face: str,
        xs: int, zs: int, xe: int, ze: int) -> Tuple[int, int, int]:
        '''
        To place block (x, y, z), which block should we target at?
        '''
        x1, y1, z1 = None, None, None
        if face == 'top':
            x1 = x
            y1 = y - 1
            z1 = z
        elif face == 'bottom':
            x1 = x
            y1 = y + 1
            z1 = z
        elif face == 'front': # we're looking back
            y1 = y
            if xe > xs:
                x1 = x - 1
                z1 = z
            elif xe < xs:
                x1 = x + 1
                z1 = z
            elif ze > zs:
                z1 = z - 1
                x1 = x
            else:
                z1 = z + 1
                x1 = x
        return x1, y1, z1

    def search_pitch(self, x_target, y_target, z_target,
        yaw: float, face: str, xdiff: bool, sneaking: bool=False):
        '''
        xdiff:
            True: the player is looking along the x axis.
            False: the player is looking along the z axis.
        '''
        logger.info(f'Trying to target at block '
            f'{(x_target, y_target, z_target)}...')
        PLAYER_EYE_LEVEL = 1.62 # see https://minecraft.wiki/w/Player#Trivia
        if sneaking:
            PLAYER_EYE_LEVEL -= 0.3

        # First, we guess the pitch from the default eye level of the player
        # and where the player should look at.
        x0, y0, z0 = self.xyz
        yp = y0 + PLAYER_EYE_LEVEL
        xf, yf, zf = x_target, y_target, z_target # should focus on which position
        if face == 'top':
            yf += 1
        elif face == 'bottom':
            yf += 0
        else:
            yf += .5
        if xdiff:
            xf += .5
        else:
            zf += .5
        numerator = yp - yf
        if xdiff:
            denominator = abs(x0 - xf)
        else:
            denominator = abs(z0 - zf)
        true_pitch = \
            round(math.atan2(numerator, denominator) * 180 / math.pi * 10) / 10
        logger.info(f'Search for a good pitch, target: '
            f'{(x_target, y_target, z_target)}, current: '
            f'{(x0, y0, z0)}, should focus on: '
            f'{(xf, yf, zf)}, numerator={numerator:.3f}, '
            f'denominator={denominator:.3f}...')
        assert -90 <= true_pitch <= 90
        for step in self.turn_to(yaw, true_pitch):
            yield None
        self.require_targeted_block = True
        for _ in range(4):
            yield None # wait longer, prevent lag
        self.require_targeted_block = False

        # If we cannot target at the desired block this way, we do exhaustive
        # search over the pitch, by halving the increment value step by step.
        if self.targeted_block != (x_target, y_target, z_target):
            # do exhaustive search
            logger.info(f'self.targeted_block = {self.targeted_block}, != '
                f'{(x_target, y_target, z_target)}.'
                f'Exhaustive pitch searching...')
            ok = False
            delta_pitch = 20.
            while not ok and delta_pitch > 1.:
                logger.info(f'Using delta_pitch = {delta_pitch:.1f}...')
                pitch = 89.9
                while not ok and pitch >= -89.9:
                    for step in self.turn_to(yaw, pitch):
                        yield None
                    self.require_targeted_block = True
                    yield None
                    self.require_targeted_block = False
                    if self.targeted_block == (x_target, y_target, z_target):
                        ok = True
                        true_pitch = pitch
                        break
                    pitch -= delta_pitch
                    pitch = round(pitch * 10) / 10
                delta_pitch /= 2.
        logger.info('Pitch searching done.')
        yield true_pitch
    
    def move_to(self, x, y, z, epsilon=0.3, yaw_tol=5.):
        '''
        epsilon: allowed error
        yaw_tol: tolerance for error of yaw
        '''
        logger.info(f'Moving to ({x:.3f}, {y:.5f}, {z:.3f})...')
        self.require_targeted_block = False
        is_sneaking = False

        # First, we do coarse adjustment by running/walking.
        coarse_eps = 5 * epsilon
        if abs(self.xyz[0] - x) > coarse_eps or \
            abs(self.xyz[1] - y) > coarse_eps or \
            abs(self.xyz[2] - z) > coarse_eps:
            logger.info('move_to: Start coarse adjustment...')
            start_moving_forward(sprint=self.allow_sprint,
                interval=self.interval)
            while abs(self.xyz[0] - x) > coarse_eps or \
                abs(self.xyz[1] - y) > coarse_eps or \
                abs(self.xyz[2] - z) > coarse_eps:
                yaw = self.get_yaw(x, z, self.xyz[0], self.xyz[2])
                if yaw is None:
                    break
                if abs(yaw - self.direction[0]) > yaw_tol:
                    stop_moving_forward() # when turning, stop moving
                    for _ in range(3):
                        yield # wait longer
                    for step in self.turn_to(yaw, self.direction[1]):
                        yield
                    start_moving_forward(sprint=self.allow_sprint and \
                        not is_sneaking, interval=self.interval)
                    continue
                if not is_sneaking and math.hypot(
                        x - self.xyz[0], y - self.xyz[1], z - self.xyz[2]
                    ) < epsilon:
                    # start sneaking
                    is_sneaking = True
                    start_shift()
                    continue
                yield
            stop_moving_forward() # but shift may still be pressed (or must be?)

        # Next, we do fine adjustment by sneaking for a very short period.
        FINE_ADJUSTMENT_SLEEPING_PERIOD = 0.01
        logger.info('move_to: Start fine adjustment...')
        do_sleep = True
        while abs(self.xyz[0] - x) > epsilon or \
            abs(self.xyz[1] - y) > epsilon or \
            abs(self.xyz[2] - z) > epsilon:
            yaw = self.get_yaw(x, z, self.xyz[0], self.xyz[2])
            if yaw is None:
                break
            if abs(yaw - self.direction[0]) > yaw_tol:
                for step in self.turn_to(yaw, self.direction[1]):
                    yield
            before = time.perf_counter_ns()
            start_moving_forward(sprint=False)
            if do_sleep:
                time.sleep(FINE_ADJUSTMENT_SLEEPING_PERIOD)
            stop_moving_forward()
            after = time.perf_counter_ns()
            # print(after - before)
            if do_sleep and after - before > FINE_ADJUSTMENT_SLEEPING_PERIOD * 1.2e9:
                do_sleep = False
                logger.info('Will not sleep in fine adjustment.')
            for _ in range(10):
                yield # wait longer

        if is_sneaking:
            stop_shift()
        logger.info('Moving done.')

    def build_line(self, task: LineBuildingTask, epsilon=0.3):
        if task.face not in ['top', 'bottom', 'front']:
            logger.error('[purple]build_line[/purple]: Invalid face name.')
            return
        xs, ys, zs = task.start # build from task.start to task.end
        xe, ye, ze = task.end
        if xs != xe and zs != ze:
            logger.error('A line must be aligned with x or z axis.')
            return
        if ys != ye:
            logger.error('Cannot build a line '
                f'with different starting and ending y\'s: {ys}, {ye}.')
            return

        ###### STAGE I: MOVING TOWARDS THE TARGET LOCATION (x1, y1, z1) ######
        x0, y0, z0 = self.xyz # current position
        line_vector = np.array([xe - xs, ye - ys, ze - zs], dtype=np.float32)
        lv_normed = line_vector / np.linalg.norm(line_vector, ord=2)
        x1, y1, z1 = xs + lv_normed[0] * self.back, ys, zs + lv_normed[2] * self.back
            # will move to (x1, y1, z1), and target at block (xs, ys, zs)
        x1 += .5 # go to the center of the block
        z1 += .5
        if task.face != 'top':
            y1 -= task.y_inc
        yaw = self.get_yaw(x1, z1, x0, z0)
        if yaw is None:
            yaw = self.direction[0]
        pitch = 0.
        for step in self.turn_to(yaw, pitch):
            yield
        for step in self.move_to(x1, y1, z1, epsilon=epsilon):
            yield

        ###### STAGE II: SEARCHING FOR A GOOD ORIENTATION ######
        x0, y0, z0 = self.xyz
        if xe > xs:
            yaw = 90 # looking towards negative x
        elif xe < xs:
            yaw = -90 # looking towards positive x
        elif ze > zs:
            yaw = -180 # looking towards negative z
        else:
            yaw = 0 # looking towards positive z
        xt, yt, zt = self.translate_block(xs, ys, zs, task.face,
            xs, zs, xe, ze)
        pitch = None
        xdiff = xe != xs
        if task.sneak:
            start_shift()
            time.sleep(1)
        for step in self.search_pitch(
            xt, yt, zt, yaw, task.face, xdiff, sneaking=task.sneak):
            if step is not None:
                pitch = step
            yield

        ###### STAGE III: DO BUILDING ######
        cur_target_block = xs, ys, zs
        last_targeted_block = None

        def finished():
            nonlocal cur_target_block
            if xe > xs:
                return cur_target_block[0] >= xe
            elif xe < xs:
                return cur_target_block[0] <= xe
            elif ze > zs:
                return cur_target_block[2] >= ze
            else:
                return cur_target_block[2] <= ze
        
        def next_target_block():
            nonlocal cur_target_block
            xc, yc, zc = cur_target_block
            if xe > xs:
                xc += 1 + task.spacing
            elif xe < xs:
                xc -= 1 + task.spacing
            elif ze > zs:
                zc += 1 + task.spacing
            else:
                zc -= 1 + task.spacing
            cur_target_block = xc, yc, zc

        n_placed = 0 # number of blocks placed.
        start_moving_backward()
        while not finished():
            xc, yc, zc = cur_target_block
            xt, yt, zt = self.translate_block(xc, yc, zc,
                task.face, xs, zs, xe, ze)
            last_targeted_block = xt, yt, zt
            self.require_targeted_block = True
            while self.targeted_block != (xt, yt, zt):
                yield
                xu, zu = self.targeted_block[0], self.targeted_block[2]
                if (xe > xs and xu > xt) or (xe < xs and xu < xt) or \
                    (ze > zs and zu > zt) or (ze < zs and zu < zt) or \
                    (xe != xs and abs(zu - zt) > epsilon) or \
                    (ze != zs and abs(xu - xt) > epsilon):
                    # missed the block, or deviate too much horizontally
                    stop_moving_backward()
                    xp, yp, zp = (
                        xc + lv_normed[0] * self.back + .5,
                        yc,
                        zc + lv_normed[2] * self.back + .5
                    )
                    if task.face != 'top':
                        yp -= task.y_inc
                    for step in self.move_to(
                        xp, yp, zp, epsilon=epsilon):
                        yield
                    for step in self.turn_to(yaw, pitch):
                        yield
                    start_moving_backward()
            self.require_targeted_block = False
            pyautogui.rightClick()
            n_placed += 1
            if n_placed % 64 == 0:
                stop_moving_backward()
                start_moving_forward()
                while self.targeted_block != last_targeted_block:
                    yield
                stop_moving_forward()
                pyautogui.middleClick()
                time.sleep(0.5)
                start_moving_backward()
            next_target_block()
        stop_moving_backward()
        if task.sneak:
            stop_shift()
        logger.info(f'Task of building line accomplished; placed '
            f'{n_placed} blocks.')

    def build_area(self, task: AreaBuildingTask):
        pass

    def finite_state_machine(self):
        '''
        When it needs game state info, it yields.
        '''
        while True:
            if self.task_queue.empty():
                self.is_idle = True
                yield
            else:
                self.is_idle = False
                task = self.task_queue.get()
                if isinstance(task, LineBuildingTask):
                    for step in self.build_line(task):
                        yield
                elif isinstance(task, AreaBuildingTask):
                    for step in self.build_area(task):
                        yield

    async def operate(self, fsm): # fsm: generator
        '''
        A single step.
        '''
        try:
            time_stamps = [time.time_ns()]
            regions = capture_window(self.w,
                regions=self.textpos.gather(self.FIELDS))
            self.xyz = self.extractor.extract_xyz(regions[0])
            towards_mask = scale_image(
                self.extractor.templates['(Towards'].candidates[0].mask,
                self.extractor.gui_scale)
            row, col = match_template(
                cv2.cvtColor(regions[1], cv2.COLOR_BGR2GRAY),
                towards_mask,
                pixel_value=self.extractor.pixel_value,
                threshold=self.extractor.threshold)
            self.direction = self.extractor.extract_direction(regions[1],
                location=(row, col + towards_mask.shape[1]))
            if self.require_targeted_block:
                self.targeted_block = self.extractor.extract_targeted_block(
                    regions[2])
            # print(self.xyz, self.direction, self.targeted_block)
            time_stamps.append(time.time_ns())
            next(fsm)
            time_stamps.append(time.time_ns())
            # print((time_stamps[-1] - time_stamps[0]) / 1e6, 'ms')
        except KeyboardInterrupt:
            pass
        except KeyError:
            logger.error('Please target at a block for now.')
            self.try_setup()
        except:
            traceback.print_exc()
            logger.info('Extracting text from clipped regions failed, rerun '
                'AutoBuilder.setup()...')
            self.try_setup()

    async def operating_thread_loop(self):
        fsm = self.finite_state_machine()
        while not self.interrupted.is_set():
            try:
                time_before = time.perf_counter_ns()
                if not self.is_idle or not self.task_queue.empty():
                    operate_task = asyncio.create_task(self.operate(fsm))
                    # sleep_task = asyncio.create_task(
                    #     asyncio.sleep(self.gap / 1000))
                    await operate_task
                time_operate = time.perf_counter_ns()
                sleep_interval = \
                    (self.gap - (time_operate - time_before) / 1e6) / 1e3
                if sleep_interval > 0.01: # 10ms
                    time.sleep(sleep_interval)
                time_after = time.perf_counter_ns()
            except:
                pass
    
    def operating_thread_func(self):
        self.require_targeted_block = False
        asyncio.run(self.operating_thread_loop())

    def run(self):
        # capture the first image. this might take somewhat long...
        self.try_setup(critical=True)

        self.opthread = threading.Thread(target=self.operating_thread_func,
            args=())
        self.opthread.start()

        with keyboard.GlobalHotKeys({
            self.interrupt_hotkey: self.set_interrupted}) as hotkey_thread:
            self.hotkey_thread = hotkey_thread
            self.hotkey_thread.join()
        
        self.opthread.join()