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
import logger
import numba

'''
The following predictors predict the possible
'''

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
    sym_num_pixels = [sym.shape[0] * sym.shape[1] for sym in symbols]
    column = 0
    while column < w:
        # best_match = -1
        # largest_num_pixels = 0
        # for i, sym in enumerate(symbols):
        #     if column + sym.shape[1] <= w:
        #         # portion of matched pixels in terms of
        #         # number of pixels in sym
        #         img_slice = img[:sym.shape[0],
        #             column:column + sym.shape[1]]
        #         matched = (img_slice == sym).sum()
        #         score = matched / sym_num_pixels[i]
        #         if score > threshold and \
        #             sym_num_pixels[i] > largest_num_pixels:
        #                 largest_num_pixels = sym_num_pixels[i]
        #                 best_match = i
        # if best_match == -1:
        #     column += jump
        # else:
        #     result.append(best_match)
        #     column += symbols[best_match].shape[1] + jump
        for i, sym in enumerate(symbols):
            if column + sym.shape[1] <= w:
                # portion of matched pixels in terms of
                # number of pixels in sym
                img_slice = img[:sym.shape[0],
                    column:column + sym.shape[1]]
                matched = (img_slice == sym).sum()
                score = matched / sym_num_pixels[i]
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
    returns (row, column) or None
    '''
    template = boolean_template.astype(np.uint8) * pixel_value
    match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    if max_val >= threshold:
        x, y = max_loc
        return y, x
    else:
        return None

class TextExtractor:
    def __init__(self, threshold=0.98, pixel_value=221, gui_scale=None):
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
                    f'{quote(s)}.npy')
                tem = Template(s, np.load(filename))
                r.append(tem)
            templates[key] = FieldInfo(f.required, r)
        return templates
    
    def load_symbols(self):
        syms = '0123456789.:/,()'
        symlist = np.array([ord(c) for c in syms])
        symbols = dict()
        for s in syms:
            if s == '/':
                filename = os.path.join(template_image_dir,
                    f'slash.npy')
            else:
                filename = os.path.join(template_image_dir,
                    f'{quote(s)}.npy')
            symbols[s] = np.load(filename)
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
                    mask = mask.repeat(scale, axis=0) \
                               .repeat(scale, axis=1)
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

    def guess_gui_scale(self, img: np.ndarray):
        '''
        img here must be an image of the whole client area.
        '''
        # plan: guess scale only when user doesn't specify scale
        for scale in [4, 1, 2, 3]:
            found, results = self.match_templates(img, scale)
            if found:
                self.gui_scale = scale
                self.scaled_symbols = dict()
                for k, v in self.symbols.items():
                    self.scaled_symbols[k] = \
                        v.repeat(scale, axis=0).repeat(scale, axis=1)
                return scale, results
        return None
    
    def preprocess(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        if location is not None:
            if size is None:
                img = img[location[0]:, location[1]:]
            else:
                img = img[location[0]:location[0] + size[0],
                    location[1]:location[1] + size[1]]

        if img.dtype == np.uint8:
            img = (img == self.pixel_value)
            if len(img.shape) > 2:
                img = img.all(aixs=2)
        img = cv2.resize((img * 255).astype(np.uint8), dsize=(
                img.shape[1] // self.gui_scale,
                img.shape[0] // self.gui_scale
            ), interpolation=cv2.INTER_LINEAR) > 127
        return img

    def extract_xyz(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        '''
        Assume img is a single-line image with height equal to
        that of the scaled XYZ templates.
        '''
        img = self.preprocess(img, location, size)

        chars = '0123456789/:.'
        symbols = numba.typed.List([self.symbols[c] for c in chars])
        idx_list = parse_symbols(img, symbols, self.threshold, 1)
        string = ''.join([chars[i] for i in idx_list])
        if string.startswith(':'):
            string = string[1:]
        result = tuple(map(float, string.split('/')))
        assert len(result) == 3
        return result

    def extract_direction(self, img: np.ndarray,
        location: Tuple[int, int]=None,
        size: Tuple[int, int]=None):
        img = self.preprocess(img, location, size)

        chars = '0123456789/:.'
        symbols = numba.typed.List([self.symbols[c] for c in chars])
        idx_list = parse_symbols(img, symbols, self.threshold, 1)
        string = ''.join([chars[i] for i in idx_list])
        if string.startswith(':'):
            string = string[1:]
        result = tuple(map(float, string.split('/')))
        assert len(result) == 3
        return result