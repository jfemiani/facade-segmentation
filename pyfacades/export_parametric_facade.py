import os

import numpy as np
import yaml
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import binary_erosion
from skimage.morphology import binary_opening
from skimage.morphology import disk
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects

import pyfacades.driving_12x360x480.segment as driving
import pyfacades.independant_12_layers.segment as facade
from pyfacades.util import channels_first
from rectify import Homography


def _guess_color(colors, n_iter=20, std=0.01):
    colors = colors.reshape(-1, 3)
    masked = (colors == (0, 0, 0)).all(1)
    colors = colors[~masked]

    best_mu = np.array([0, 0, 0])
    best_n = 0
    if len(colors) == 0:
        return best_mu
    for k in range(n_iter):
        subset = colors[np.random.choice(np.arange(len(colors)), 10)]
        mu = subset.mean(0)
        inliers = (((colors - mu) ** 2 / std) < 1).all(1)

        mu = colors[inliers].mean(0)
        n = len(colors[inliers])
        if n > best_n:
            best_n = n
            best_mu = mu
    return best_mu


def guess_color(chunks, i, n_iter=20, std=0.01):
    colors = chunks[i].reshape(-1, 3)
    return _guess_color(colors, n_iter, std)


def find_facade_cuts(facade_sig):
    run, start, val = rle(facade_sig > facade_sig.mean() + facade_sig.std())
    result = []
    for s, e in zip(start[val], start[val] + run[val]):
        result.append(s + facade_sig[s:e].argmax())
    result = [0] + result + [len(facade_sig) - 1]
    result = np.unique(result)
    return np.array(result)


def rle(inarray):
    """ run length encoding"""
    ia = np.array(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def get_boxes(image, threshold=0.5, se=disk(3)):
    mask = image > threshold
    mask = binary_opening(mask, selem=se)
    try:
        props = regionprops(label(mask))

        def _tag(tlbr):
            t, l, b, r = tlbr
            return dict(top=int(t), left=int(l), bottom=int(b), right=int(r))
        result = [_tag(r.bbox) for r in props]
    except (ValueError, TypeError) as e:
        result = []
    return result


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

    def load_image(self, path, **kwargs):
        self.data = imread(path)
        self.path = path



        # Sometimes an approximate mask can be produced based on Google range data
        # the mask indicates which parts of the image are not facade
        mask_path = os.path.join(os.path.dirname(self.path), 'mask.png')
        if os.path.isfile(mask_path):
            self.data_mask = rgb2gray(imread(mask_path)) > 0.5
        else:
            self.data_mask = None

        # We try to rectify prior to classification to normalize the input.
        # This can be a cause of failure....
        self.homography = Homography(self.data, mask=self.data_mask)
        self.rectified = self.homography.rectified
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


def _process_strip(strip_index, cuts, sky_strips, door_strips, win_strips, dbg=None):
    # Sky / Roof
    sky_sig = np.median(sky_strips[strip_index], axis=1)
    sky_sig /= max(1e-4, sky_sig.max())

    sky_rows = np.where(sky_sig > 0.5)[0]
    if len(sky_rows) > 0:
        roof_top = sky_rows[-1]
    else:
        roof_top = 0

    # Door
    door_sig = np.percentile(door_strips[strip_index], 75, axis=1)
    door_sig /= max(1e-4, door_sig.max())
    door_thresh = door_sig.mean() + door_sig.std()
    door_top = np.where(door_sig > door_thresh)[0]
    if door_top is None or len(door_top) == 0:
        door_top = len(door_sig)
    else:
        door_top = door_top.min()

    # HACK ALERT: Windows can be mistaken for doors, and that thros the whole thing off.
    #             As a solution, we reject a solution that puts the doors more than
    #             300 pixels up from the bottom
    if len(door_sig) - door_top > 300:
        door_top = len(door_sig)

    # Windows
    win_sig = np.percentile(win_strips[strip_index], 85, axis=1)
    win_sig[sky_sig > 0.5] = 0
    if win_sig.max() > 0:
        win_sig /= win_sig.max()

    # Windows Summary
    #  -- initialize (some wont be set on all paths through th code)
    win_heights = []
    win_widths = []

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
        win_bottom = win_top = win_vspacing = win_height = -1

    win_hsig = None
    if len(win_heights) > 0:
        win_hsig = np.percentile(win_strips[strip_index][win_top:win_bottom], 85, axis=0)
        runs, starts, values = rle(win_hsig > 0.5)
        starts += cuts[strip_index]
        win_widths = runs[np.atleast_1d(values)]
        win_widths = np.atleast_1d(win_widths)
        win_lefts = np.atleast_1d(starts[values])
        if len(win_widths) > 0:
            win_left = win_lefts[0]
            win_right = win_lefts[-1] + win_widths[-1]
            win_hspacing = np.diff(win_lefts).mean() if len(win_lefts) > 1 else 0
            #win_width = win_widths.mean()
        else:
            win_left = win_right = win_hspacing = -1 #win_width = -1
    else:
        win_widths = win_lefts = []
        win_left = win_right = win_hspacing = -1

    meta = dict()
    meta['facade-left'] = int(cuts[strip_index])
    meta['facade-right'] = int(cuts[strip_index + 1])
    meta['sky-line'] = int(roof_top)
    meta['door-line'] = int(door_top)
    meta['win-count'] = len(win_heights) * len(win_widths)

    win_meta = dict()
    win_meta['top'] = int(win_top)
    win_meta['bottom'] = int(win_bottom)
    win_meta['left'] = int(win_left)
    win_meta['right'] = int(win_right)

    win_meta['rows'] = len(win_heights)
    win_meta['cols'] = len(win_widths)

    win_meta['height'] = int(win_heights.mean()) if len(win_heights) else -1
    win_meta['width'] = int(win_widths.mean()) if len(win_widths) else -1

    win_meta['vspacing'] = float(win_vspacing)
    win_meta['hspacing'] = float(win_hspacing)

    if len(win_widths) and len(win_heights):
        t, l = np.meshgrid(win_tops, win_lefts)
        h, w = np.meshgrid(win_heights, win_widths)

        tlbr = np.stack((t.flatten(), l.flatten(), (t + h).flatten(), (l + w).flatten()), 0).T

        win_meta['tlbr'] = tlbr.tolist()
    else:
        win_meta['tlbr'] = []
    meta['win_grid'] = win_meta

    meta['win_v_margin'] = win_sig.tolist()
    meta['win_h_margin'] = win_hsig.tolist() if win_hsig is not None else []

    return meta


