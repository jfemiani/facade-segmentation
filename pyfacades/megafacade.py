import os

import yaml
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, disk

from pyfacades.driving_12x360x480 import segment as driving
from pyfacades.export_parametric_facade import find_facade_cuts, _process_strip, _guess_color, get_boxes
from pyfacades.independant_12_layers import segment as facade
from pyfacades.rectify import Homography
from pyfacades.util import channels_first


class FacadeStrip(object):
    def __init__(self):
        super(FacadeStrip, self).__init__()

    def to_dict(self):
        pass


class MegaFacade(object):
    def __init__(self):
        super(MegaFacade, self).__init__()
        self.data = None
        self.path = ''
        self.rectified = None
        self.homography = None
        self.window_scores = None
        self.sky_mask = None
        self.facade_edge_scores = None
        self.data_mask = None
        self.sky_feature_map = None
        self.facade_feature_map = None
        self.params = []

    def _mask_out_common_obstructions(self):
        """Mask out the sky and some common objects that can obstruct a sky"""
        pass
    
    def load_image(self, path, **kwargs):
        self.data = imread(path)
        self.path = path
        self._load_image_mask()
        self._mask_out_common_obstructions()
        self._rectify_image()

        self.sky_feature_map = driving.process_strip(channels_first(self.rectified * 255))
        self.facade_feature_map, conf = facade.process_strip(channels_first(self.rectified * 255))

        self.window_scores = facade.model.window(self.facade_feature_map)
        if self.data_mask is not None:
            self.window_scores[~self.data_mask] = 0

        # Sky segmentation is poor / noisy, so use some morph ops to 'fix' it hackishly
        self.sky_mask = driving.model.sky(self.sky_feature_map) > 0.5
        remove_small_holes(self.sky_mask, min_size=20 * 20, in_place=True)
        remove_small_objects(self.sky_mask, min_size=20 * 20, in_place=True)

        # Ignore anything that seems to be sky
        self.window_scores[self.sky_mask] = 0

        # Ignore anything that seems to be NOT part of a building
        self.window_scores[driving.occlusion(self.sky_feature_map)] = 0

        # For facades, ignore the sky but do not use the mask (facade edges may be masked out)
        self.facade_edge_scores = facade.model.facade_edge(self.facade_feature_map)
        self.facade_edge_scores[self.sky_mask] = 0
        self.facade_edge_scores[facade.model.door(self.facade_feature_map) > 0.5] = 0
        self.facade_edge_scores[self.window_scores > 0.5] = 0

        facade_sig = self.facade_edge_scores.sum(0)
        facade_cuts = find_facade_cuts(facade_sig)

        facade_mask = driving.model.building(self.sky_feature_map)
        facade_mask = binary_erosion(facade_mask, disk(10))  # Sky is noisy

        # Remove non-wall elements from the facade (we want just the wall)
        facade_mask[self.window_scores > 0.5] = 0
        facade_mask[facade.model.door(self.facade_feature_map) > 0.5] = 0
        facade_mask[facade.model.balcony(self.facade_feature_map) > 0.5] = 0
        facade_mask[facade.model.shop(self.facade_feature_map) > 0.5] = 0
        facade_mask[facade.model.pillar(self.facade_feature_map) > 0.5] = 0
        facade_mask[facade.model.molding(self.facade_feature_map) > 0.5] = 0

        wall_colors = self.rectified.copy()
        wall_colors[~facade_mask, :] = 0, 0, 0

        self.params = self.process_strips(wall_colors, facade_cuts)

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

        rgb_strips = [stripped[:, s:e] for (s, e) in zip(cuts[:-1], cuts[1:])]
        win_strips = [self.window_scores[:, s:e].copy() for (s, e) in zip(cuts[:-1], cuts[1:])]
        door_strips = [door_scores[:, s:e].copy() for (s, e) in zip(cuts[:-1], cuts[1:])]
        sky_strips = [self.sky_mask[:, s:e].copy() for (s, e) in zip(cuts[:-1], cuts[1:])]

        # Noise of some sort seems to happen near the edges of strips
        for s in win_strips + sky_strips:
            s[:, :1] = s[:, -1:] = 0  # edge effects

        strips = []
        for strip_index in range(len(cuts) - 1):
            meta = _process_strip(strip_index, cuts, sky_strips, door_strips, win_strips)

            intersting_patch = stripped[meta['sky-line']:meta['door-line'], meta['facade-left']:meta['facade-right'], :]

            color = _guess_color(intersting_patch)
            meta['rgb'] = color.tolist()

            left, right = meta['facade-left'], meta['facade-right']
            facade_strip = self.facade_feature_map[:, :, :, left:right]
            driving_strip = self.sky_mask[:, left:right]
            meta['regions'] = {}
            meta['regions']['window'] = get_boxes(facade.model.window(facade_strip))
            meta['regions']['door'] = get_boxes(facade.model.door(facade_strip))
            meta['regions']['balcony'] = get_boxes(facade.model.balcony(facade_strip))
            meta['regions']['sill'] = get_boxes(facade.model.sill(facade_strip))
            meta['regions']['cornice'] = get_boxes(facade.model.cornice(facade_strip))
            meta['regions']['shop'] = get_boxes(facade.model.shop(facade_strip))

            strips.append(meta)

        return strips

    def save_params(self, path=None,  **kwargs):

        if path is None:
            path = os.path.join(os.path.dirname(self.path), 'parameters.yml')

        with open(path, 'wb') as f:
            meta = dict()
            meta['facades'] = self.params
            meta['homography'] = dict(inv=self.homography.inv_H.tolist(),
                                      tfm=self.homography.H.tolist())
            f.write(yaml.dump(meta))