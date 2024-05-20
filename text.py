template_image_dir = './templates'

f'''
Files required: {template_image_dir}/*
'''

from collections import namedtuple
from urllib.parse import quote
from typing import Iterable, List, Tuple
import numpy as np
import cv2
import os
import imageio.v3 as iio
import logger
import numba


# available symbol templates
syms = '0123456789-.:/,()pn'


@numba.njit(cache=True)
def parse_symbols(img: np.ndarray,
    symbols: numba.typed.List,
    threshold: float=0.98,
    jump: int=1):
    '''
    Parse the text from a single-line 0/1 image: img.
    symbols: a list of symbols (templates) to be parsed.
    jump: the number of pixels between symbols.

    Note: we want ., :, ... to have the lowest priority
    so place them at the back of the symbols list.
    '''
    w = img.shape[1]
    result = []
    column = 0
    while column < w:
        for i, sym in enumerate(symbols):
            if column + sym.shape[1] <= w:
                # score: portion of matched pixels in terms of
                # number of pixels in sym
                img_slice = img[:sym.shape[0],
                    column:column + sym.shape[1]]
                matched = (img_slice == sym).sum()
                score = matched / sym.size
                if score > threshold:
                    result.append(i)
                    column += symbols[i].shape[1] + jump - 1
                    break
        column += 1
    return np.array(result, dtype=np.int32)

def match_template(img: np.ndarray,
    boolean_template: np.ndarray,
    pixel_value: int=221,
    threshold=0.98):
    '''
    img: grayscale image.
    returns (row, column) or None
    '''
    template = boolean_template.astype(np.uint8)
    match = cv2.matchTemplate((img == pixel_value).astype(np.uint8),
        template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if max_val >= threshold:
        x, y = max_loc
        return y, x
    else:
        return None
    
def scale_image(img: np.ndarray, scale: int):
    return img.repeat(scale, axis=0).repeat(scale, axis=1)

class TextExtractor:
    def __init__(self, threshold: float=0.98,
        pixel_value: int=221, gui_scale: int=None):
        self.threshold = threshold
        self.templates = self.load_templates()
        self.symlist, self.symbols = self.load_symbols()
        self.pixel_value = pixel_value
        self.gui_scale = gui_scale

    def load_templates(self):
        FieldInfo = namedtuple('FieldInfo', ['required', 'candidates'])
        Template = namedtuple('Template', ['text', 'mask'])
        template_dict = {
            'XYZ': FieldInfo(True, ['XYZ', 'XYZ:']),
            '(Towards': FieldInfo(True, ['(Towards']),
            'Client Light': FieldInfo(False,
                ['Client Light', 'Client Light:']),
            'Targeted Block': FieldInfo(False,
                ['Targeted Block', 'Targeted Block:']),
            'Targeted Fluid': FieldInfo(False,
                ['Targeted Fluid', 'Targeted Fluid:'])
        } # 'True' here means absolutely critical informatiom
        # that must be retrieved in match_templates
        templates = {}
        for key, f in template_dict.items():
            r = []
            l = f.candidates
            for s in l:
                filename = os.path.join(template_image_dir,
                    f'{quote(s)}.png')
                tem = Template(s, iio.imread(filename) > 0)
                r.append(tem)
            templates[key] = FieldInfo(f.required, r)
        return templates
    
    def load_symbols(self):
        symlist = np.array([ord(c) for c in syms])
        symbols = dict()
        for s in syms:
            if s == '/':
                filename = os.path.join(template_image_dir,
                    f'slash.png')
            else:
                filename = os.path.join(template_image_dir,
                    f'{quote(s)}.png')
            symbols[s] = iio.imread(filename) > 0
        return symlist, symbols

    def match_templates(self, img: np.ndarray,
        scale=1, required: Iterable=set()):
        '''
        e.g. required = {'Targeted Block', 'Targeted Fluid'}
        '''
        if not isinstance(required, set):
            required = set(required)
        results = dict()
        found = False
        for k, v in self.templates.items():
            if v.required or k in required or not found:
                # if found, then skip non-required fields
                for cand in v.candidates:
                    mask = cand.mask
                    mask = scale_image(mask, scale)
                    MatchingResult = namedtuple('MatchingResult',
                        ['coordinate', 'mask_size'])
                    result = match_template(img, mask,
                        pixel_value=self.pixel_value,
                        threshold=self.threshold)
                    if result is not None:
                        found = True
                        results[k] = MatchingResult(result, mask.shape)
                        break
        return found, results

    def guess_gui_scale(self, img: np.ndarray, required: Iterable=set()):
        '''
        img here must be an image of the whole client area.
        '''
        # plan: guess scale only when user doesn't specify scale
        for scale in [4, 1, 2, 3, 5]:
            found, results = self.match_templates(img, scale, required)
            if found:
                self.gui_scale = scale
                self.scaled_symbols = dict()
                for k, v in self.symbols.items():
                    self.scaled_symbols[k] = scale_image(v, scale)
                return scale, results
        return None
    
    def preprocess(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None,
        resize_image=False):
        '''
        resize_image: resize the image to like gui_scale=1 in order to
            accelerate text extraction, but may affect accuracy.
        '''
        if location is not None:
            if size is None:
                img = img[location[0]:, location[1]:]
            else:
                img = img[location[0]:location[0] + size[0],
                    location[1]:location[1] + size[1]]

        if img.dtype == np.uint8:
            img = (img == self.pixel_value)
            if len(img.shape) > 2:
                img = img.all(axis=2)

        if resize_image:
            nh, nw = img.shape[0] // self.gui_scale, \
                img.shape[1] // self.gui_scale
            h, w = nh * self.gui_scale, nw * self.gui_scale
            img = img[:h, :w]
            img = (img.astype(np.float32)).reshape(
                (nh, self.gui_scale, nw, self.gui_scale)) \
                .mean(axis=1).mean(axis=2) > .5
        return img

    def extract_text(self, img: np.ndarray, chars: str,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None,
        resize_image=False,
        do_preprocess=True):
        '''
        resize_image: resize the image to like gui_scale=1 in order to
            accelerate text extraction, but may affect accuracy.
            If False, then use scaled symbols.
        '''
        if do_preprocess:
            img = self.preprocess(img, location, size, resize_image)
        symbols = numba.typed.List([
            self.symbols[c] if resize_image else self.scaled_symbols[c]
            for c in chars])
        idx_list = parse_symbols(img, symbols, self.threshold, 1)
        string = ''.join([chars[i] for i in idx_list])
        return string

    def extract_xyz(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        '''
        Assume img is a single-line image with height equal to
        that of the scaled XYZ templates.
        '''
        string = self.extract_text(img=img, chars='0123456789-/:.',
            location=location, size=size)
        # remove non-digit characters in prefix and suffix
        for s in string:
            if s.isdigit() or s =='-':
                break
            string = string[1:]
        for s in reversed(string):
            if s.isdigit() or s == '-':
                break
            string = string[:-1]
        try:
            result = tuple(map(float, string.split('/')))
        except:
            raise RuntimeError(f'Error occurred in extract_xyz: string={string}')
        assert len(result) == 3
        return result

    def extract_direction(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        
        img = self.preprocess(img, location, size)

        matched = lambda x, y: (x == y).sum() / y.size > self.threshold

        sym_p, sym_n, sym_lbr = self.scaled_symbols['p'], \
            self.scaled_symbols['n'], self.scaled_symbols['(']
        vert = max([self.scaled_symbols[c].shape[0] for c in 'pn()'])

        if matched(img[:sym_p.shape[0], self.gui_scale * 5: \
            self.gui_scale * 5 + sym_p.shape[1]], sym_p) and \
            matched(img[:sym_lbr.shape[0], self.gui_scale * 61: \
            self.gui_scale * 61 + sym_lbr.shape[1]], sym_lbr):
            # towards positive X/Z
            img = img[:vert, self.gui_scale * 61 + sym_lbr.shape[1]:]
        elif matched(img[:sym_n.shape[0], self.gui_scale * 5: \
            self.gui_scale * 5 + sym_n.shape[1]], sym_n) and \
            matched(img[:sym_lbr.shape[0], self.gui_scale * 65: \
            self.gui_scale * 65 + sym_lbr.shape[1]], sym_lbr):
            # towards negative X/Z
            img = img[:vert, self.gui_scale * 65 + sym_lbr.shape[1]:]

        string = self.extract_text(img=img, chars='0123456789-/.()',
            location=location, size=size, do_preprocess=False)
        result = []
        negative = False
        cur_num = 0
        decimal = False
        base = 1.
        for ch in string:
            if ch.isdigit():
                if decimal:
                    base /= 10
                    cur_num += int(ch) * base
                else:
                    cur_num = cur_num * 10 + int(ch)
            elif ch == '.':
                decimal = True
            elif ch == '/':
                result.append(-cur_num if negative else cur_num)
                negative = False
                cur_num = 0
                decimal = False
                base = 1.
            elif ch == '-':
                negative = True
        result.append(-cur_num if negative else cur_num)
        result = tuple(result)
        assert len(result) == 2, \
            f'extract_direction: len(result) != 2, result={result}'
        return result

    def extract_targeted_block(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        string = self.extract_text(img=img, chars='0123456789-,',
            location=location, size=size)
        try:
            result = tuple(map(int, string.split(',')))
        except:
            raise RuntimeError(f'Error occurred in extract_targeted_block: '
                f'string={string}')
        assert len(result) == 3
        return result