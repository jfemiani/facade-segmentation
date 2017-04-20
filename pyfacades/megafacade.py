import os

import numpy as np
import yaml
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, disk

from pyfacades.driving_12x360x480 import segment as driving
from pyfacades.export_parametric_facade import find_facade_cuts, _guess_color, get_boxes, rle
from pyfacades.independant_12_layers import segment as facade
from pyfacades.rectify import Homography
from pyfacades.util import channels_first


class WindowGrid(object):
    def __init__(self, door_top, roof_top, sky_sig, win_strip, facade_left):
        super(WindowGrid, self).__init__()

        self.top = -1
        self.bottom = -1
        self.left = -1
        self.right = -1
        self.vertical_signal = []
        self.horizontal_signal = []
        self.tops = []
        self.heights = []
        self.lefts = []
        self.widths = []
        self.vertical_spacing = 0
        self.horizontal_spacing = 0

        self._cut_windows_vertically(door_top, roof_top, sky_sig, win_strip)
        self._cut_windows_horizontally(facade_left, win_strip)

        self.rows = len(self.heights)
        self.cols = len(self.widths)
        self.window_height = int(self.heights.mean()) if len(self.heights) else -1
        self.window_width = int(self.widths.mean()) if len(self.widths) else -1

        if self.rows and self.cols:
            t, l = np.meshgrid(self.tops, self.lefts)
            h, w = np.meshgrid(self.heights, self.widths)
            rectangles = np.stack((t.flatten(), l.flatten(), (t + h).flatten(), (l + w).flatten()), 0).T
            self.rectangles = rectangles.tolist()
        else:
            self.rectangles = []

    def _cut_windows_horizontally(self, s, win_strip):
        win_hsig = []
        if len(self.heights) > 0:
            win_hsig = np.percentile(win_strip[self.top:self.bottom], 85, axis=0)
            runs, starts, values = rle(win_hsig > 0.5)
            starts += s
            win_widths = runs[np.atleast_1d(values)]
            win_widths = np.atleast_1d(win_widths)
            win_lefts = np.atleast_1d(starts[values])
            if len(win_widths) > 0:
                win_left = win_lefts[0]
                win_right = win_lefts[-1] + win_widths[-1]
                win_hspacing = np.diff(win_lefts).mean() if len(win_lefts) > 1 else 0
                # win_width = win_widths.mean()
            else:
                win_left = win_right = win_hspacing = -1  # win_width = -1
        else:
            win_widths = win_lefts = []
            win_left = win_right = win_hspacing = -1

        self.horizontal_spacing = win_hspacing
        self.left = win_left
        self.right = win_right
        self.horizontal_signal = win_hsig
        self.lefts = win_lefts
        self.widths = win_widths

    def _cut_windows_vertically(self, door_top, roof_top, sky_sig, win_strip):
        win_sig = np.percentile(win_strip, 85, axis=1)
        win_sig[sky_sig > 0.5] = 0
        if win_sig.max() > 0:
            win_sig /= win_sig.max()
        win_sig[:roof_top] = 0
        win_sig[door_top:] = 0
        runs, starts, values = rle(win_sig > 0.5)
        win_heights = runs[values]
        win_tops = starts[values]
        if len(win_heights) > 0:
            win_bottom = win_tops[-1] + win_heights[-1]
            win_top = win_tops[0]
            win_vspacing = np.diff(win_tops).mean() if len(win_tops) > 1 else 0
        else:
            win_bottom = win_top = win_vspacing = -1

        self.top = win_top
        self.bottom = win_bottom
        self.vertical_spacing = win_vspacing
        self.vertical_signal = win_sig
        self.heights = win_heights
        self.tops = win_tops

    def to_dict(self):
        meta = dict()
        meta['top'] = self.top
        meta['bottom'] = self.bottom
        meta['left'] = self.left
        meta['right'] = self.right
        meta['rows'] = self.rows
        meta['cols'] = self.cols
        meta['height'] = self.window_height
        meta['width'] = self.window_width
        meta['vertical_spacing'] = self.vertical_spacing
        meta['horizontal_spacing'] = self.horizontal_spacing
        meta['rectangles'] = self.rectangles
        return meta


class FacadeStrip(object):
    def __init__(self, facade_left, facade_right, sky_strip, door_strip, win_strip, dbg=None):
        super(FacadeStrip, self).__init__()

        # Sky / Roof
        sky_sig = np.median(sky_strip, axis=1)
        sky_sig /= max(1e-4, sky_sig.max())

        sky_rows = np.where(sky_sig > 0.5)[0]
        if len(sky_rows) > 0:
            roof_top = sky_rows[-1]
        else:
            roof_top = 0

        # Door
        door_sig = np.percentile(door_strip, 75, axis=1)
        door_sig /= max(1e-4, door_sig.max())
        door_thresh = door_sig.mean() + door_sig.std()
        door_top = np.where(door_sig > door_thresh)[0]
        if door_top is None or len(door_top) == 0:
            door_top = len(door_sig)
        else:
            door_top = door_top.min()

        # HACK ALERT: Windows can be mistaken for doors, and that throws the whole thing off.
        #             As a solution, we reject a solution that puts the doors more than
        #             300 pixels up from the bottom
        if len(door_sig) - door_top > 300:
            door_top = len(door_sig)

        # Windows

        self.window_grid = WindowGrid(door_top, roof_top, sky_sig, win_strip, facade_left)
        self.facade_left = int(facade_left)
        self.facade_right = int(facade_right)
        self.sky_line = int(roof_top)
        self.door_line = int(door_top)
        self.win_count = self.window_grid.rows*self.window_grid.cols

    def to_dict(self):
        meta = dict()
        meta['facade-left'] = self.facade_left
        meta['facade_right'] = self.facade_right
        meta['sky_line'] = self.sky_line
        meta['door_line'] = self.door_line
        meta['win_count'] = self.win_count
        meta['window_grid'] = self.window_grid.to_dict()
        meta['win_v_margin'] = list(self.window_grid.vertical_signal)
        meta['win_h_margin'] = list(self.window_grid.horizontal_signal)
        return meta


class MegaFacade(object):
    def __init__(self):
        super(MegaFacade, self).__init__()
        self.data = None
        self.path = ''
        self.rectified = None
        self.homography = None
        self.window_scores = None
        self._sky_mask = None
        self.facade_edge_scores = None
        self.data_mask = None
        self.sky_feature_map = None
        self.facade_feature_map = None
        self.params = []

    def _mask_out_common_obstructions(self):
        """Mask out the sky and some common objects that can obstruct a facade 
        
        This is intended to be run prior to rectifications, since these things can prevent us from 
        correctly identifying the rectilinear features of a facade
        
        """
        pass

    def load_image(self, path, **kwargs):
        self.data = imread(path)
        self.path = path
        self._load_image_mask()
        self._mask_out_common_obstructions()
        self._rectify_image()

        self.sky_feature_map = driving.process_strip(channels_first(self.rectified * 255))
        self.facade_feature_map, conf = facade.process_strip(channels_first(self.rectified * 255))

        self._create_sky_mask()
        self._segment_windows()
        self._segment_facade_edges()

        facade_cuts = self._split_at_facade_edges()
        facade_mask = self._create_facade_mask()
        wall_colors = self._create_wall_colors(facade_mask)

        self.params = self.process_strips(wall_colors, facade_cuts)

    def _create_wall_colors(self, facade_mask):
        wall_colors = self.rectified.copy()
        wall_colors[~facade_mask, :] = 0, 0, 0
        return wall_colors

    def _create_facade_mask(self):
        facade_mask = self.building_mask()
        facade_mask = binary_erosion(facade_mask, disk(10))  # Sky is noisy
        # Remove non-wall elements from the facade (we want just the wall)
        facade_mask[self.window_mask()] = 0
        facade_mask[self.door_mask()] = 0
        facade_mask[self.balcony_mask()] = 0
        facade_mask[self.shop_mask()] = 0
        facade_mask[self.pillar_mask()] = 0
        facade_mask[self.molding_mask()] = 0
        return facade_mask

    def sky_mask(self):
        return self._sky_mask

    def building_mask(self):
        return driving.model.building(self.sky_feature_map) > 0.5

    def molding_mask(self):
        return facade.model.molding(self.facade_feature_map) > 0.5

    def pillar_mask(self):
        return facade.model.pillar(self.facade_feature_map) > 0.5

    def shop_mask(self):
        return facade.model.shop(self.facade_feature_map) > 0.5

    def balcony_mask(self):
        return facade.model.balcony(self.facade_feature_map) > 0.5

    def window_mask(self):
        return self.window_scores > 0.5

    def _split_at_facade_edges(self):
        facade_sig = self.facade_edge_scores.sum(0)
        facade_cuts = find_facade_cuts(facade_sig)
        return facade_cuts

    def _segment_facade_edges(self):
        # For facades, ignore the sky but do not use the mask (facade edges may be masked out)
        self.facade_edge_scores = facade.model.facade_edge(self.facade_feature_map)
        self.facade_edge_scores[self._sky_mask] = 0
        self.facade_edge_scores[self.door_mask()] = 0
        self.facade_edge_scores[self.window_mask()] = 0

    def door_mask(self):
        return facade.model.door(self.facade_feature_map) > 0.5

    def _segment_windows(self):
        # Ignore anything that seems to be sky
        self.window_scores = facade.model.window(self.facade_feature_map)
        if self.data_mask is not None:
            self.window_scores[~self.data_mask] = 0
        self.window_scores[self._sky_mask] = 0
        # Ignore anything that seems to be NOT part of a building
        self.window_scores[driving.occlusion(self.sky_feature_map)] = 0

    def _create_sky_mask(self):
        # Sky segmentation is poor / noisy, so use some morph ops to 'fix' it hackishly
        self._sky_mask = driving.model.sky(self.sky_feature_map) > 0.5
        remove_small_holes(self._sky_mask, min_size=20 * 20, in_place=True)
        remove_small_objects(self._sky_mask, min_size=20 * 20, in_place=True)

    def _rectify_image(self):
        # We try to rectify prior to classification to normalize the input.
        # This can be a cause of failure....
        self.homography = Homography(self.data, mask=self.data_mask)
        self.rectified = self.homography.rectified

    def _load_image_mask(self):
        # Sometimes an approximate mask can be produced based on Google range data
        # the mask indicates which parts of the image are not facade
        mask_path = os.path.join(os.path.dirname(self.path), 'mask.png')
        if os.path.isfile(mask_path):
            self.data_mask = rgb2gray(imread(mask_path)) > 0.5
        else:
            self.data_mask = None

    def process_strips(self, stripped, cuts):
        door_scores = facade.model.door(self.facade_feature_map)

        strip_cuts = zip(cuts[:-1], cuts[1:])
        strips = []
        for strip_index in range(len(strip_cuts)):
            s, e = strip_cuts[strip_index]
            meta = self._create_mini_facade(cuts, door_scores, e, s, strip_index, stripped)
            strips.append(meta)

        return strips

    def _create_mini_facade(self, cuts, door_scores, e, s, strip_index, stripped):
        door_strip = door_scores[:, s:e].copy()
        win_strip = self.window_scores[:, s:e].copy()
        sky_strip = self._sky_mask[:, s:e].copy()
        win_strip[:, :1] = win_strip[:, -1:] = 0  # edge effects
        sky_strip[:, :1] = sky_strip[:, -1:] = 0  # edge effects

        facade = FacadeStrip(s, e, sky_strip, door_strip, win_strip)
        meta = facade.to_dict()
        interesting_patch = stripped[meta['sky-line']:meta['door-line'], meta['facade-left']:meta['facade-right'], :]
        color = _guess_color(interesting_patch)

        meta['rgb'] = color.tolist()
        left, right = meta['facade-left'], meta['facade-right']
        facade_strip = self.facade_feature_map[:, :, :, left:right]
        meta['regions'] = {}
        meta['regions']['window'] = get_boxes(facade.model.window(facade_strip))
        meta['regions']['door'] = get_boxes(facade.model.door(facade_strip))
        meta['regions']['balcony'] = get_boxes(facade.model.balcony(facade_strip))
        meta['regions']['sill'] = get_boxes(facade.model.sill(facade_strip))
        meta['regions']['cornice'] = get_boxes(facade.model.cornice(facade_strip))
        meta['regions']['shop'] = get_boxes(facade.model.shop(facade_strip))
        return meta

    def save_params(self, path=None,  **kwargs):

        if path is None:
            path = os.path.join(os.path.dirname(self.path), 'parameters.yml')

        with open(path, 'wb') as f:
            meta = dict()
            meta['facades'] = self.params
            meta['homography'] = dict(inv=self.homography.inv_H.tolist(),
                                      tfm=self.homography.H.tolist())
            f.write(yaml.dump(meta))

