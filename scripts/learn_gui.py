#!/usr/bin/env python

from __future__ import print_function

# TODO:
#  - create file from command line with named colors
#  - random train/test split of tiles
#  - remove color from command line
#  - better interactive tools / bootstrapping
#  - non-rectangular area?
#
# DONE:
#  - DONE: save last view in dot file?
#  - DONE: brute-force nearest neighbor for unlabeled pixels?
#  - DONE: customizable per-file blur sigma to cleanup
#  - DONE: spit out blobfinder2 data file
#  - DONE: replace pylons with cones in lab
#  
# UPDATED ROS NODE
#  - just 4 topics:
#    - debug_image
#    - blobs
#    - blobs3d
#  - arg or param for colors
#   - 
#  

import sys
import os
import cv2
import numpy as np
import json
import scipy.ndimage

import re

CHAN_BITS = np.array([5, 6, 5])

TOTAL_BITS = CHAN_BITS.sum()

LUT_SIZE = (1 << TOTAL_BITS)

MAX_COLORS = 8

EXTENSIONS = ['.png', '.jpg']

WINDOW = 'Color picker'

MODES = ['draw', 'roundtrip', 'classified', 'error', 'all_classified']

POS_COLOR = np.array([0, 255, 127], dtype=np.uint8)
BUF_COLOR = np.array([255, 127, 0], dtype=np.uint8)

WHICH_COLORS = dict(pos=POS_COLOR, buf=BUF_COLOR)

TRUE_POS_COLOR = np.array([0, 127, 0], dtype=np.uint8)
TRUE_NEG_COLOR = np.array([0, 0, 127], dtype=np.uint8)
FALSE_POS_COLOR = np.array([0, 0, 255], dtype=np.uint8)
FALSE_NEG_COLOR = np.array([0, 255, 0], dtype=np.uint8)

PRED_BLUR_SIGMA = 1.4 # chosen to overfit to minimize error

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
LINE_HEIGHT = 18
TAB_WIDTH = 200

POS_WEIGHT = 0.5 # lower to reduce false positive values
NEG_WEIGHT = 1.0 - POS_WEIGHT

RECT_TOL = 2

HANDLE_TOL = 8

HELP = '''Keyboard controls (* denotes in draw mode only)

  Q or ESC\tquit

  [ or ]\tchange current image
  < or >\tchange current mode
  X or C\tchange current color

  A or -\tzoom out
  Z or +\tzoom in 
  SHIFT + CLICK\tscroll

  P\tadd positive rectangle*
  B\tadd buffer rectangle*
  T\ttoggle selected rectangle type*
  D or DEL\tdelete selected rectangle*
  TAB\tselect next rectangle

  ?\tdisplay this screen

Press any key to continue...
'''


JSON_REPS = [
    (r'\[\n *([0-9]+)', '[ \\1'),
    (r'([^.])([0-9]+,?) *\n *', '\\1\\2 ')
]


HANDLES = np.array([[0.5, 1.0], # bottom
                    [1.0, 0.5], # right
                    [0.5, 0.0], # top
                    [0.0, 0.5], # left
                    [0.0, 0.0], # top left
                    [1.0, 0.0], # top right
                    [1.0, 1.0], # bottom right
                    [0.0, 1.0],]) # bottom left
              
HANDLE_MASKS = np.array([
    [0, 0, 0, 1], # bottom
    [0, 0, 1, 0], # right
    [0, 1, 0, 0], # top
    [1, 0, 0, 0], # left
    [1, 1, 0, 0], # top left
    [0, 1, 1, 0], # top right
    [0, 0, 1, 1], # bottom right
    [1, 0, 0, 1], # bottom left
    [1, 1, 1, 1], # entire rectangle
], dtype=bool)


def apply_delta(cur_value, num_values, delta):
    return (cur_value + delta) % num_values

def put_text(img, text, pos, scale):

    for (brt, width) in [(0, 3), (255, 1)]:

        cv2.putText(img, text, pos, FONT_FACE, scale,
                    (brt, brt, brt), width, cv2.LINE_AA)

def rect_dist(rect, pt):

    (x0, y0, x1, y1) = rect

    x, y = pt

    return max(y - y1, max(x - x1, max(x0 - x, y0 - y)))


def constrain(x0, x1, w, a0, a1):

    if a0:
        if a1:
            cmin = -x0
            cmax = w - x1
        else:
            cmin = -x0
            cmax = x1 - x0 - 1
    elif a1:
        cmin = x0 - x1 + 1
        cmax = w - x1
    else:
        cmin = 0
        cmax = 0

    return (cmin, cmax)

def to_indexed(img):

    h, w, nchan = img.shape
        
    assert nchan == len(CHAN_BITS)
    assert TOTAL_BITS <= 16

    indexed = np.zeros((h, w), dtype=np.uint8)

    for chan, bits in enumerate(CHAN_BITS):

        lshift = CHAN_BITS[chan+1:].sum()

        img_chan = img[:, :, chan].astype(np.uint16) 

        rshift = (8 - bits) 
        
        rshift_mask = img_chan >> rshift

        lshifted = rshift_mask << lshift

        indexed = indexed | lshifted


    return indexed

def from_indexed(indexed):

    h, w = indexed.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)

    for chan, bits in enumerate(CHAN_BITS):

        rshift = CHAN_BITS[chan+1:].sum()
        lshift = 8 - bits
        mask = (1 << bits) - 1
        offs = 1 << (lshift - 1)

        img[:, :, chan] = (((indexed >> rshift) & mask) << lshift) + offs

    return img
        
def roundtrip(img):

    return from_indexed(to_indexed(img))

    #if COLORSPACE != -1:
    #img = cv2.cvtColor(img, COLORSPACE)

    #if COLORSPACE_INV != -1:
    #    img = cv2.cvtColor(img, COLORSPACE_INV)

    #return img

def cleanup(mask, sigma):

    if sigma == 0.0:
        return mask

    #mask = np.where(mask, np.uint8(255), np.uint8(0))
    mask = mask * 255

    blur = cv2.GaussianBlur(mask, (0, 0), sigma)

    return (blur > 127).view(np.uint8)

    #mask = np.where(blur > 127, np.uint8(255), np.uint8(0))



class LearnGUI:

    def __init__(self, dirname):

        self.dirname = dirname

        self.images = []

        self.image_dims = dict()

        max_dims = np.array([0, 0])

        for filename in sorted(os.listdir(dirname)):
            _, ext = os.path.splitext(filename)
            if ext in EXTENSIONS:
                fullfile = os.path.join(dirname, filename)
                img = cv2.imread(fullfile)
                if img is not None:
                    self.images.append(filename)
                    dims = img.shape[:2]
                    self.image_dims[filename] = dims
                    max_dims = np.maximum(max_dims, dims)

        if len(self.images) == 0:
            print('no images found in', dirname)
            sys.exit(1)

        print('got images:', self.images)
        print('max dims are', max_dims)


        h, w = max_dims
        self.display = np.zeros((h, w, 3), dtype=np.uint8)

        x = np.arange(w)
        y = np.arange(h)

        x, y = np.meshgrid(x, y)
        
        hstripe = x % 2 != 0
        vstripe = y % 2 != 0

        self.checker = np.where(hstripe ^ vstripe, np.uint8(255), np.uint8(0))
        self.checker = self.checker.view(bool)

        self.display_help = False

        self.mouse_mode = None
        self.mouse_point = None
        self.handle_point = None

        self.pred_blur_sigma = PRED_BLUR_SIGMA

        self.load_data()

        self.train()

        self.save_lut()

        flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE

        cv2.namedWindow(WINDOW, flags)


        image_index, color_index, mode_index = self.load_config()


        self.set_mode(image_index, color_index, mode_index)

    

    def train(self):

        ncolors = len(self.colors)

        pn_counts = np.zeros((ncolors, LUT_SIZE, 2), dtype=np.int32)

        self.lut = np.zeros(LUT_SIZE, dtype=np.uint8)

        for filename in self.images:

            img = cv2.imread(os.path.join(self.dirname, filename))

            indexed = to_indexed(img)

            for cidx, color in enumerate(self.colors):

                masks = self.get_masks(filename, color)

                pos_idx = indexed[masks['pos'].view(bool)]
                neg_idx = indexed[masks['neg'].view(bool)]

                num_pos = len(pos_idx)
                num_neg = len(neg_idx)

                if num_pos == 0:
                    print('skipping color {} for image {} because no positive examples'.format(
                        color, filename))
                    continue

                pos_idx, pos_counts = np.unique(pos_idx, return_counts=True)
                neg_idx, neg_counts = np.unique(neg_idx, return_counts=True)

                assert pos_counts.sum() == num_pos
                assert neg_counts.sum() == num_neg
                
                pn_counts[cidx, pos_idx, 0] += pos_counts
                pn_counts[cidx, neg_idx, 1] += neg_counts

        print()

        lut_3d_shape = 1 << CHAN_BITS

        delta_bits = CHAN_BITS.max() - CHAN_BITS
        grid_spacing = (1 << delta_bits)

        for cidx, color in enumerate(self.colors):

            num_pos, num_neg = pn_counts[cidx].sum(axis=0)

            print()
            print('color {} has {} pos, {} neg'.format(color, num_pos, num_neg))
            
            if num_pos == 0:
                print('skipping color {} because no samples'.format(color))
                continue

            w_pos = POS_WEIGHT / num_pos
            w_neg = NEG_WEIGHT / num_neg

            w = np.array([w_pos, w_neg])
                
            w_counts = pn_counts[cidx] * w

            is_pos = (w_counts[:,0] > w_counts[:,1]).astype(np.uint8)

            is_unlabeled = (pn_counts[cidx].sum(axis=1) == 0)

            print('  got {}/{} ({:.2f}%) unlabled indices for color {}'.format(
                is_unlabeled.sum(), LUT_SIZE, 
                100.0*is_unlabeled.sum()/LUT_SIZE, color))

            bin_image = is_unlabeled.reshape(lut_3d_shape)

            print('  computing the fancy expensive EDT...')

            dists, indices = scipy.ndimage.distance_transform_edt(
                bin_image, 
                sampling=grid_spacing, 
                return_distances=True, return_indices=True)

            print('  dists is', dists.shape, dists.dtype)
            print('  indices is', indices.shape, indices.dtype)

            indices = indices.reshape((len(CHAN_BITS), LUT_SIZE))

            closest_ind = np.zeros(LUT_SIZE, dtype=np.int32)

            for chan in range(len(CHAN_BITS)):
                lshift = CHAN_BITS[chan+1:].sum()
                closest_ind[:] = closest_ind | (indices[chan, :] << lshift)

            is_labeled, = np.nonzero(~is_unlabeled)

            assert np.all(closest_ind[is_labeled] == is_labeled)

            print('  before EDT correction, is_pos.sum() =', is_pos.sum())
            
            is_pos[is_unlabeled] = is_pos[closest_ind[is_unlabeled]]
            
            print('  after EDT correction, is_pos.sum() =', is_pos.sum())


            self.lut = self.lut | (is_pos << cidx)

        self.evaluate()

    def evaluate(self):

        ncolors = len(self.colors)

        total_error = np.zeros(ncolors, dtype=np.float64)
        total_count = np.zeros(ncolors, dtype=np.float64)
        total_color = np.zeros((ncolors, 3), dtype=np.float64)
        total_pcount = np.zeros(ncolors, dtype=np.float64)

        for filename in self.images:

            img = cv2.imread(os.path.join(self.dirname, filename))

            indexed = to_indexed(img)
            all_pred = self.lut[indexed]

            for cidx, color in enumerate(self.colors):
                
                masks = self.get_masks(filename, color)

                pos_mask = masks['pos'].view(bool)
                neg_mask = masks['neg'].view(bool)

                num_pos = pos_mask.sum()
                num_neg = neg_mask.sum()

                num_total = num_pos + num_neg

                if num_pos == 0:
                    continue

                w_pos = POS_WEIGHT / num_pos
                w_neg = NEG_WEIGHT / num_neg

                pred = ((all_pred >> cidx) & 1)

                pred = cleanup(pred, self.pred_blur_sigma).view(bool)

                error = ( w_pos * (pos_mask & ~pred).sum() +
                          w_neg * (neg_mask & pred).sum() )

                total_error[cidx] += num_total * error
                total_count[cidx] += num_total

                total_color[cidx] += num_pos * img[pos_mask].mean(axis=0)
                total_pcount[cidx] += num_pos

        self.mean_colors = np.full((ncolors, 3), 127, dtype=np.uint8)

        print()
                
        for cidx, color in enumerate(self.colors):

            if total_count[cidx] == 0:
                continue

            error = total_error[cidx] / total_count[cidx]
            print('error for color {} is {:.7f}'.format(color, error))

            mc = np.clip(total_color[cidx] / total_pcount[cidx], 0, 255).astype(np.uint8)

            self.mean_colors[cidx] = mc

        print()


        if total_count.sum():
            error = total_error.sum() / total_count.sum()
            print('overall error is {:.7f}'.format(error))
            
            

                

    def mouse(self, event, x, y, flags, _):

        if self.display_help: 
            return

        mode = MODES[self.cur_mode_index]

        p = np.array([x, y])

        if self.mouse_mode == 'scroll':

            if event == cv2.EVENT_LBUTTONUP:

                self.mouse_mode = None

            else:

                delta = p - self.mouse_point

                self.mouse_point = p

                self.cur_center -= delta / self.cur_scale

                self.draw()

        elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_SHIFTKEY:

            self.mouse_mode = 'scroll'
            self.mouse_point = p

        if mode != 'draw':
            return

        filename = self.images[self.cur_image_index]

        color = self.colors[self.cur_color_index]

        which_rects = self.get_rectangles(filename, color)

        if self.mouse_mode == 'drag':

            if event == cv2.EVENT_LBUTTONUP:

                self.mouse_mode = None

            else:

                key, idx = self.cur_rect
                
                delta = p - self.mouse_point

                offs = (delta / self.cur_scale).astype(int)

                active = HANDLE_MASKS[self.cur_handle_index]

                x0, y0, x1, y1 = self.orig_rect

                ih, iw = self.image_dims[filename]

                dx_min, dx_max = constrain(x0, x1, iw, active[0], active[2])
                dy_min, dy_max = constrain(y0, y1, ih, active[1], active[3])

                offs = np.clip(offs, (dx_min, dy_min), (dx_max, dy_max))

                offs = np.hstack((offs, offs))

                offs[~active] = 0

                new_rect = self.orig_rect + offs

                which_rects[key][idx] = new_rect

                self.draw()
                

        elif event == cv2.EVENT_LBUTTONDOWN:


            if self.cur_rect is not None:

                key, idx = self.cur_rect

                rect = which_rects[key][idx]

                handles = self.get_rect_handles(rect)

                hdiff = handles - p

                hdist = np.abs(hdiff).sum(axis=1)
                
                # prioritize last
                hidx = 7 - hdist[::-1].argmin()
                hmin = hdist[hidx]

                if hmin < HANDLE_TOL:

                    self.mouse_mode = 'drag'

                    self.cur_handle_index = hidx

                    self.mouse_point = p
                    self.orig_rect = rect.copy()

                else:

                    xformed_rect = self.transform_rect(rect)

                    rdist = rect_dist(xformed_rect, p)

                    if rdist < RECT_TOL:

                        self.mouse_mode = 'drag'

                        self.cur_handle_index = -1

                        self.mouse_point = p
                        self.orig_rect = rect.copy()


            if self.mouse_mode is None:

                for key in ['pos', 'buf']:

                    for idx, rect in enumerate(which_rects[key]):

                        xformed_rect = self.transform_rect(rect)

                        rdist = rect_dist(xformed_rect, p)

                        if rdist < RECT_TOL:
                            self.mouse_mode = 'drag'
                            self.cur_handle_index = -1
                            self.mouse_point = p
                            self.orig_rect = rect.copy()
                            self.cur_rect = (key, idx)
                            self.draw()
                            break

                    if self.mouse_mode is not None:
                        break
                        
            if self.mouse_mode is None:
                self.cur_rect = None
                self.draw()

            # then active rect
            # then non active rects

            pass

    def set_mode(self, 
                 image_index=None, color_index=None, mode_index=None, 
                 delta_image_index=None, delta_color_index=None, delta_mode_index=None):

        image_updated = False

        if image_index is not None:
            self.cur_image_index = image_index
            image_updated = True

        if delta_image_index is not None:
            self.cur_image_index = apply_delta(self.cur_image_index, len(self.images), delta_image_index)
            image_updated = True

        if color_index is not None:
            self.cur_color_index = color_index

        if delta_color_index is not None:
            self.cur_color_index = apply_delta(self.cur_color_index, len(self.colors), delta_color_index)
            
        if mode_index is not None:
            self.cur_mode_index = mode_index

        if delta_mode_index is not None:
            self.cur_mode_index = apply_delta(self.cur_mode_index, len(MODES), delta_mode_index)

        if image_updated:

            filename = self.images[self.cur_image_index]
            h, w = self.image_dims[filename]

            self.cur_center = np.array([w//2, h//2], dtype=np.float32)
            self.cur_scale = 1

            self.cur_image = cv2.imread(os.path.join(self.dirname, filename))
            
            kx = np.array([[0.25, 0.5, 0.25]])
            ky = kx.transpose()

            self.cur_image = cv2.sepFilter2D(self.cur_image,
                                             cv2.CV_8U,
                                             kx, ky)
                                             

        self.cur_rect = None
        self.cur_handle_index = -1
        
        self.draw()

    def transform_rect(self, rect):

        (x0, y0, x1, y1) = rect - np.float32(0.5)

        (x0, y0, _) = np.ceil(np.dot(self.cur_xform, 
                                     np.array([x0, y0, 1]))).astype(int)

        (x1, y1, _) = np.ceil(np.dot(self.cur_xform, 
                                     np.array([x1, y1, 1]))).astype(int)

        return np.array([x0, y0, x1, y1])

    def get_rect_handles(self, rect):

        (x0, y0, x1, y1) = self.transform_rect(rect)

        p0 = np.array([x0, y0])
        p1 = np.array([x1, y1])

        p0 = p0[None, :]
        p1 = p1[None, :] - 1

        p = (p0 * (1 - HANDLES) + p1 * (HANDLES)).astype(int)

        return p

        

    def draw(self):

        filename = self.images[self.cur_image_index]
        color = self.colors[self.cur_color_index]
        mode = MODES[self.cur_mode_index]

        h, w = self.display.shape[:2]

        # transform image center to origin
        x0, y0 = self.cur_center

        T0 = np.array([[1.0, 0.0, -x0],
                       [0.0, 1.0, -y0],
                       [0.0, 0.0, 1.0]])

        s = self.cur_scale

        T1 = np.array([[s, 0.0, 0.0],
                       [0.0, s, 0.0],
                       [0.0, 0.0, 1.0]])


        T2 = np.array([[1.0, 0.0, w//2],
                       [0.0, 1.0, h//2],
                       [0.0, 0.0, 1.0]])

        self.cur_xform = np.dot(T2, np.dot(T1, T0))

        if mode == 'draw':

            which_rects = self.get_rectangles(filename, color)


            masks = self.get_masks(filename, color)

            cv2.warpAffine(self.cur_image, self.cur_xform[:2],
                           (w, h), self.display, cv2.INTER_NEAREST)

            pmask = cv2.warpAffine(masks['pos'], self.cur_xform[:2],
                           (w, h), None, cv2.INTER_NEAREST)\
                       .view(bool)

            bmask = cv2.warpAffine(masks['buf'], self.cur_xform[:2],
                                   (w, h), None, cv2.INTER_NEAREST)\
                       .view(bool)

            overlap = pmask & bmask

            pmask[overlap & self.checker] = False
            bmask[overlap & ~self.checker] = False

            self.display[pmask] = self.display[pmask] // 2 + POS_COLOR // 2
            self.display[bmask] = self.display[bmask] // 2 + BUF_COLOR // 2

            if self.cur_rect is not None:

                key, idx = self.cur_rect

                rect = which_rects[key][idx]

                lite = tuple([int(ci) for ci in WHICH_COLORS[key]])
                dark = tuple([int(ci//2) for ci in WHICH_COLORS[key]])

                
                handles = self.get_rect_handles(rect)

                for x, y in handles:

                    cv2.rectangle(self.display, 
                                  (x-2, y-2), (x+2, y+2),
                                  dark, -1)

                    cv2.rectangle(self.display, 
                                  (x-1, y-1), (x+1, y+1),
                                  lite, -1)

        elif mode == 'roundtrip':

            img = roundtrip(self.cur_image)

            cv2.warpAffine(img, self.cur_xform[:2],
                           (w, h), self.display, cv2.INTER_NEAREST)

        elif mode == 'classified' or mode == 'error' or mode == 'all_classified':


            indexed = to_indexed(self.cur_image)
            all_pred = self.lut[indexed]

            showme = np.full((h, w, 3), 127, dtype=np.uint8)


            if mode == 'all_classified':

                pmask = np.zeros((h, w), dtype=bool)

                for cidx in range(len(self.colors)):

                    pred = ((all_pred >> cidx) & 1)

                    pred = cleanup(pred, self.pred_blur_sigma).view(bool)

                    pmask[pred] = True
                    showme[pred] = self.mean_colors[cidx]

            else:

                pred = ((all_pred >> self.cur_color_index) & 1)

                pred = cleanup(pred, self.pred_blur_sigma).view(bool)

                if mode == 'classified':

                    showme[pred] = self.mean_colors[self.cur_color_index]

                else:

                    masks = self.get_masks(filename, color)

                    pos_mask = masks['pos'].view(bool)
                    neg_mask = masks['neg'].view(bool)

                    if not np.any(pos_mask):
                        pos_mask[:] = False
                        neg_mask[:] = False

                    assert (pos_mask & neg_mask).sum() == 0

                    showme[pos_mask & pred] = TRUE_POS_COLOR
                    showme[neg_mask & pred] = FALSE_POS_COLOR

                    showme[pos_mask & ~pred] = FALSE_NEG_COLOR
                    showme[neg_mask & ~pred] = TRUE_NEG_COLOR

            cv2.warpAffine(showme, self.cur_xform[:2],
                           (w, h), self.display, cv2.INTER_NEAREST)
            
            

        else:

            self.display[:] = 127

        ##################################################

        if self.display_help:

            self.display //= 2
            
            y = LINE_HEIGHT*2

            for line in HELP.split('\n'):
                x = LINE_HEIGHT
                if '\t' in line:
                    for text in line.split('\t'):
                        put_text(self.display, text, (x, y), FONT_SCALE)
                        tabstop = x // TAB_WIDTH
                        x = TAB_WIDTH * (tabstop + 1)
                else:
                    put_text(self.display, line, (x, y), FONT_SCALE)
                y += LINE_HEIGHT

        else:
            
            x = LINE_HEIGHT // 2
            y = LINE_HEIGHT 
            put_text(self.display, 'Image: {}; Color: {}; Mode: {}'.format(
                filename, color, mode), (x, y), FONT_SCALE)

        cv2.imshow(WINDOW, self.display)
        
    def load_config(self):


        dirname = os.path.expanduser('~')

        filename = os.path.join(dirname, '.colorlut2')

        if not os.path.exists(filename):
            return 0, 0, 0

        image_index = 0
        color_index = 0
        mode_index = 0

        with open(filename, 'r') as istr:

            dv = json.load(istr)

            if 'image' in dv:
                try:
                    image_index = self.images.index(dv['image'])
                except:
                    pass

            if 'color' in dv:
                try: 
                    color_index = self.colors.index(dv['color'])
                except:
                    pass

            if 'mode' in dv:
                try:
                    mode_index = MODES.index(dv['mode'])
                except:
                    pass

        return image_index, color_index, mode_index


    def load_data(self):

        jsonfile = os.path.join(self.dirname, 'colordata.json')

        was_changed = False

        if not os.path.exists(jsonfile):
            
            self.colors = ['default_color']
            
            self.rectangles = dict()

            self.save_data()

        else:
            
            with open(jsonfile, 'r') as istr:
                jsdata = json.load(istr)
                
            self.colors = jsdata['colors']
            self.rectangles = dict()

            jsr = jsdata['rectangles']

            for filename, jsc in jsr.items():
                c = dict()
                for color, jst in jsc.items():
                    t = dict()
                    for key, jsa in jst.items():
                        if len(jsa):
                            a = np.array(jsa)
                            assert len(a.shape) == 2
                            assert a.shape[1] == 4
                            t[key] = a
                    c[color] = t
                self.rectangles[filename] = c

            try:
                self.pred_blur_sigma = jsdata['pred_blur_sigma']
            except KeyError:
                self.pred_blur_sigma = PRED_BLUR_SIGMA

    def save_config(self):

        color = self.colors[self.cur_color_index]
        image = self.images[self.cur_image_index]
        mode = MODES[self.cur_mode_index]

        dirname = os.path.expanduser('~')
        filename = os.path.join(dirname, '.colorlut2')

        config = dict(color=color, image=image, mode=mode)

        with open(filename, 'w') as ostr:
            json.dump(config, ostr, indent=2, sort_keys=True)

    def save_data(self):

        jsonfile = os.path.join(self.dirname, 'colordata.json')

        jsr = dict()

        for filename, c in self.rectangles.items():
            jsc = dict()
            for color, t in c.items():
                jst = dict()
                for key, a in t.items():
                    if len(a) > 0:
                        assert len(a.shape) == 2
                        assert a.shape[1] == 4
                        assert a.shape[0] > 0
                        jsa = []
                        for row in a:
                            jsa.append([int(ai) for ai in row])
                        jst[key] = jsa
                jsc[color] = jst
            jsr[filename] = jsc

        saveme = dict(colors=self.colors, rectangles=jsr, 
                      pred_blur_sigma=self.pred_blur_sigma)

        outdata = json.dumps(saveme, indent=2, sort_keys=True)

        for (pat, rep) in JSON_REPS:
            outdata = re.sub(pat, rep, outdata)

        with open(jsonfile, 'w') as ostr:
            ostr.write(outdata)
            ostr.write('\n')

    def save_lut(self):

        path = os.path.join(self.dirname, 'colorlut2.data')

        with open(path, 'w') as ostr:

            ostr.write('ColorLUT2 BGR\n')
            for bits in CHAN_BITS:
                ostr.write('{}\n'.format(bits))
            ostr.write('{}\n'.format(MAX_COLORS)) # num colors

            assert len(self.colors) <= MAX_COLORS

            for color in self.colors:
                ostr.write('{}\n'.format(color))
                
            for _ in range(MAX_COLORS - len(self.colors)):
                ostr.write('\n')

            ostr.write(repr(self.pred_blur_sigma))
            ostr.write('\n')

                
        with open(path, 'ab') as ostr:
            ostr.write(self.lut.tobytes())
            
        print('wrote', path)

    def get_rectangles(self, filename, color): # -> [pos, neg]

        null_rects = np.zeros((0, 4), dtype=np.int32)

        triv_rval = dict(pos=null_rects, buf=null_rects.copy())

        if filename not in self.rectangles:
            return triv_rval

        rdict = self.rectangles[filename]

        if color not in rdict:
            return triv_rval

        (h, w) = self.image_dims[filename]

        rdict = rdict[color]

        for key in ['pos', 'buf']:

            if key not in rdict:

                rdict[key] = null_rects.copy()

        return rdict

    def get_masks(self, filename, color):

        which_rects = self.get_rectangles(filename, color)

        masks = dict()

        for key in ['pos', 'buf']:

            rect_list = which_rects[key]

            mask = np.zeros(self.image_dims[filename], dtype=np.uint8)

            for (x0, y0, x1, y1) in rect_list:
                mask[y0:y1, x0:x1] = 255

            masks[key] = mask

        

        masks['neg'] = ~(masks['pos'] | masks['buf'])

        return masks

    def run(self):

        cv2.moveWindow(WINDOW, 0, 0)
        cv2.setMouseCallback(WINDOW, self.mouse)

        while True:

            while True:

                k = cv2.waitKey()

                if k >= 0 and self.mouse_mode is None:
                    break

            nudge = 32.0 / self.cur_scale
            redraw = False

            mode = MODES[self.cur_mode_index]


            # quit/help
            if self.display_help:
                self.display_help = False
                redraw = True

            elif k == ord('q') or k == ord('Q') or k == 27:

                self.save_data()
                self.save_config()

                return
            elif k == ord('?'):
                self.display_help = True
                redraw = True

            # navigation/zoom

            if k == ord('a') or k == ord('A') or k == ord('-'):
                self.cur_scale *= 0.5
                if self.cur_scale < 1.0:
                    self.cur_scale = 1.0
                else:
                    redraw = True
            elif k == ord('z') or k == ord('Z') or k == ord('=') or k == ord('+'):
                self.cur_scale *= 2.0
                if self.cur_scale > 16.0:
                    self.cur_scale = 16.0
                else:
                    redraw = True
            elif k == ord('0'):
                filename = self.images[self.cur_image_index]
                (h, w) = self.image_dims[filename]
                self.cur_scale = 1.0
                self.cur_center = np.array([w//2, h//2], dtype=np.float32)
                redraw = True

            # mode changing
            
            if k == ord('['):
                self.set_mode(delta_image_index = -1)
            elif k == ord(']'):
                self.set_mode(delta_image_index = 1)
            elif k == ord('<') or k == ord(','):
                self.set_mode(delta_mode_index = -1)
            elif k == ord('>') or k == ord('.'):
                self.set_mode(delta_mode_index = 1)
            elif k == ord('C') or k == ord('c'):
                self.set_mode(delta_color_index=1)
            elif k == ord('x') or k == ord('X'):
                self.set_mode(delta_color_index=-1)

            if mode == 'draw':
                if k == 9: # tab
                    self.delta_select(1)
                    redraw = True
                elif k == 11 or k == 25: # shift tab
                    self.delta_select(-1)
                    redraw = True
                elif k == ord('P') or k == ord('p'):
                    self.add_rectangle('pos')
                    redraw = True
                elif k == ord('B') or k == ord('b'):
                    self.add_rectangle('buf')
                    redraw = True
                elif k == ord('T') or k == ord('t'):
                    self.toggle_current_rectangle()
                    redraw = True
                elif k == 8 or k == 127 or k == ord('D') or k == ord('d'):
                    self.delete_current_rectangle()
                    redraw = True

            if redraw:
                self.draw()
                
    def add_rectangle(self, key, new_rect=None):

        filename = self.images[self.cur_image_index]
        color = self.colors[self.cur_color_index]
        h, w = self.image_dims[filename]

        if new_rect is None:

            xc, yc = self.cur_center.astype(int)

            r = int(32 // self.cur_scale)

            if xc < r:
                x0 = 0
                x1 = 2*r
            elif xc + r > w:
                x0 = w - 2*r
                x1 = w
            else:
                x0 = xc - r
                x1 = xc + r

            if yc < r:
                y0 = 0
                y1 = 2*r
            elif yc + r > h:
                y0 = h - 2*r
                y1 = h
            else:
                y0 = yc - r
                y1 = yc + r

            new_rect = (x0, y0, x1, y1)

        if filename not in self.rectangles:
            self.rectangles[filename] = dict()

        if color not in self.rectangles[filename]:
            self.rectangles[filename][color] = dict()

        if key not in self.rectangles[filename][color]:
            self.rectangles[filename][color][key] = np.array([new_rect])
            idx = 0
        else:
            idx = len(self.rectangles[filename][color][key])
            self.rectangles[filename][color][key] = np.vstack((
                self.rectangles[filename][color][key],
                np.array([new_rect])
            ))

        self.cur_rect = (key, idx)

    def toggle_current_rectangle(self):

        if self.cur_rect is None:
            return

        key, _ = self.cur_rect

        save_me = self.delete_current_rectangle()

        if save_me is None:
            return

        opp = dict(pos='buf', buf='pos')

        self.add_rectangle(opp[key], save_me)

    def delete_current_rectangle(self):
            
        if self.cur_rect is None:
            return

        key, idx = self.cur_rect

        filename = self.images[self.cur_image_index]
        color = self.colors[self.cur_color_index]

        which_rects = self.rectangles[filename][color]

        n = len(which_rects[key])

        ab = [idx, n-1]
        ba = [n-1, idx]

        which_rects[key][ab] = which_rects[key][ba]

        save_me = which_rects[key][n-1].copy()

        which_rects[key] = which_rects[key][:n-1]

        if len(which_rects[key]) == 0:
            del which_rects[key]

        if len(which_rects) == 0:
            del self.rectangles[filename][color]

        if len(self.rectangles[filename]) == 0:
            del self.rectangles[filename]

        self.cur_rect = None

        return save_me

    def delta_select(self, delta):

        assert delta == 1 or delta == -1


        if MODES[self.cur_mode_index] != 'draw':
            return

        filename = self.images[self.cur_image_index]
        color = self.colors[self.cur_color_index]

        which_rects = self.get_rectangles(filename, color)


        keys = ['none']
        info = dict(none=1)

        for key in ['pos', 'buf']:
            n = len(which_rects[key])
            if n > 0:
                keys.append(key)
                info[key] = n

        if self.cur_rect is None:
            key, idx = 'none', 0
        else:
            key, idx = self.cur_rect
            if key not in info or idx >= info[key]:
                key, idx = 'none', 0

        cnt = info[key]

        assert idx < cnt
        
        idx = apply_delta(idx, cnt + 1, delta)

        if idx == cnt:
            pos = keys.index(key)
            key = keys[(pos + delta) % len(keys)]
            if delta < 0:
                idx = info[key] - 1
            else:
                idx = 0

        if key == 'none':
            self.cur_rect = None
        else:
            self.cur_rect = (key, idx)
        
def main():

    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        print('usage: python {} DIRNAME'.format(os.path.sys.argv[0]))
        sys.exit(1)

    dirname = sys.argv[1]

    gui = LearnGUI(dirname)

    gui.run()
    
if __name__ == '__main__':
    main()



