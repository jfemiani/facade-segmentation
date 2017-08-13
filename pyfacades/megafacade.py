import matplotlib
import pyfacades
from scipy.ndimage import binary_closing
from skimage.measure import regionprops, label

if matplotlib.get_backend() is None:
    matplotlib.use('Agg')

import os
import pylab as pl

import numpy as np
import yaml
from os.path import join, dirname, abspath

from os.path import isfile

from scipy.misc import imsave
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.io import imread
from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, disk, binary_opening
from skimage.util.dtype import img_as_ubyte

import pyfacades.models.driving_12x360x480 as driving
import pyfacades.models.independant_12_layers as i12
from pyfacades.rectify import Homography
from pyfacades.util import channels_first


class WindowGrid(object):
    def __init__(self, door_top, roof_top, sky_sig, win_strip, facade_left):
        super(WindowGrid, self).__init__()

        self.top = -1
        self.bottom = -1
        self.left = -1
        self.right = -1
        self.vertical_scores = np.array([])
        self.horizontal_scores = np.array([])
        self.tops = np.array([])
        self.heights = np.array([])
        self.lefts = np.array([])
        self.widths = np.array([])
        self.vertical_spacing = 0
        self.horizontal_spacing = 0

        self._cut_windows_vertically(door_top, roof_top, sky_sig, win_strip)
        self._cut_windows_horizontally(facade_left, win_strip)

        self.rows = len(self.heights)
        self.cols = len(self.widths)
        self.window_height = int(np.mean(self.heights)) if len(self.heights) else -1
        self.window_width = int(np.mean(self.widths)) if len(self.widths) else -1

        if self.rows and self.cols:
            t, l = np.meshgrid(self.tops, self.lefts)
            h, w = np.meshgrid(self.heights, self.widths)
            rectangles = np.stack((t.flatten(), l.flatten(), (t + h).flatten(), (l + w).flatten()), 0).T
            self.rectangles = rectangles.tolist()
        else:
            self.rectangles = []

    def _cut_windows_horizontally(self, s, win_strip):
        win_horizontal_scores = []
        if len(self.heights) > 0:
            win_horizontal_scores = np.percentile(win_strip[self.top:self.bottom], 85, axis=0)
            runs, starts, values = run_length_encode(win_horizontal_scores > 0.5)
            starts += s
            win_widths = runs[np.atleast_1d(values)]
            win_widths = np.atleast_1d(win_widths)
            win_lefts = np.atleast_1d(starts[values])
            if len(win_widths) > 0:
                win_left = win_lefts[0]
                win_right = win_lefts[-1] + win_widths[-1]
                win_horizontal_spacing = np.diff(win_lefts).mean() if len(win_lefts) > 1 else 0
                # win_width = win_widths.mean()
            else:
                win_left = win_right = win_horizontal_spacing = -1  # win_width = -1
        else:
            win_widths = win_lefts = []
            win_left = win_right = win_horizontal_spacing = -1

        self.horizontal_spacing = int(win_horizontal_spacing)
        self.left = int(win_left)
        self.right = int(win_right)
        self.horizontal_scores = win_horizontal_scores
        self.lefts = np.array(win_lefts)
        self.widths = np.array(win_widths)

    def _cut_windows_vertically(self, door_top, roof_top, sky_sig, win_strip):
        win_sig = np.percentile(win_strip, 85, axis=1)
        win_sig[sky_sig > 0.5] = 0
        if win_sig.max() > 0:
            win_sig /= win_sig.max()
        win_sig[:roof_top] = 0
        win_sig[door_top:] = 0
        runs, starts, values = run_length_encode(win_sig > 0.5)
        win_heights = runs[values]
        win_tops = starts[values]
        if len(win_heights) > 0:
            win_bottom = win_tops[-1] + win_heights[-1]
            win_top = win_tops[0]
            win_vertical_spacing = np.diff(win_tops).mean() if len(win_tops) > 1 else 0
        else:
            win_bottom = win_top = win_vertical_spacing = -1

        self.top = int(win_top)
        self.bottom = int(win_bottom)
        self.vertical_spacing = int(win_vertical_spacing)
        self.vertical_scores = make_list(win_sig)
        self.heights = np.array(win_heights)
        self.tops = np.array(win_tops)

    def to_dict(self):
        meta = dict()
        meta['top'] = int(self.top)
        meta['bottom'] = int(self.bottom)
        meta['left'] = int(self.left)
        meta['right'] = int(self.right)
        meta['rows'] = int(self.rows)
        meta['cols'] = int(self.cols)
        meta['height'] = int(self.window_height)
        meta['width'] = int(self.window_width)
        meta['vertical_spacing'] = int(self.vertical_spacing)
        meta['horizontal_spacing'] = int(self.horizontal_spacing)
        meta['rectangles'] = make_list(self.rectangles)
        return meta

    def plot(self):
        import pylab as pl
        ax = pl.gca()

        pl.hlines(self.tops, self.left, self.right, linestyles='dashed', colors='blue')
        pl.hlines(self.tops + self.heights, self.left, self.right, linestyles='dashed', colors='green')
        pl.vlines(self.lefts, self.top, self.bottom, linestyles='dashed', colors='blue')
        pl.vlines(self.lefts + self.widths, self.top, self.bottom, linestyles='dashed', colors='green')

        for box in self.rectangles:
            t, l, b, r = box
            patch = pl.Rectangle((l, t), r - l, b - t, color='blue', fill=True, alpha=0.5)
            ax.add_patch(patch)
        pass


def make_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


class FacadeCandidate(object):
    def __init__(self, mega_facade, facade_left, facade_right, sky_strip, door_strip, win_strip, rgb_strip):
        super(FacadeCandidate, self).__init__()
        self.mega_facade = mega_facade
        #assert mega_facade is None or isinstance(self.mega_facade, MegaFacade)

        self.colors = rgb_strip

        # Some strange numpy types sometimes come in -- we want to store int values.
        facade_left = int(facade_left)
        facade_right = int(facade_right)

        # Sky / Roof
        sky_sig = np.median(sky_strip, axis=1)
        sky_sig /= max(1e-4, sky_sig.max())

        sky_rows = np.where(sky_sig > 0.5)[0]
        if len(sky_rows) > 0:
            sky_line = int(sky_rows[-1])
        else:
            sky_line = 0

        # Door
        door_sig = np.percentile(door_strip, 75, axis=1)
        # door_sig /= max(1e-4, door_sig.max())
        win_sig = np.percentile(win_strip, 75, axis=1)
        # win_sig /= max(1e-4, win_sig.max())
        door_sig = np.min((door_sig, 1 - win_sig), axis=0)

        door_thresh = door_sig.mean() + door_sig.std()
        door_line = np.where(door_sig > door_thresh)[0]
        if door_line is None or len(door_line) == 0:
            door_line = len(door_sig)
        else:
            door_line = int(door_line.min())

        # HACK ALERT: Windows can be mistaken for doors, and that throws the whole thing off.
        #             As a solution, we reject a solution that puts the doors more than
        #             300 pixels up from the bottom
        if len(door_sig) - door_line > 300:
            door_line = len(door_sig)

        # Colors
        try:
            #Sometimes the skyline is too high
            percent_up = 0.66
            top_line = int(percent_up*sky_line + (1-percent_up)*door_line)
            main_color = ransac_guess_color(rgb_strip[top_line:door_line])
        except AssertionError:
            main_color = np.array([1, 1, 1])

        try:
            mezzanine_color = ransac_guess_color(rgb_strip[door_line:])
        except AssertionError:
            mezzanine_color = main_color

        # Windows
        window_grid = WindowGrid(door_line, sky_line, sky_sig, win_strip, facade_left)

        self.color = main_color
        self.mezzanine_color = mezzanine_color
        self.window_grid = window_grid
        self.facade_left = int(facade_left)
        self.facade_right = int(facade_right)
        self.sky_line = int(sky_line)
        self.door_line = int(door_line)
        self.win_count = self.window_grid.rows * self.window_grid.cols
        self.regions = {}

    def uncertainty_for_windows(self):
        #assert isinstance(self.mega_facade, MegaFacade)
        #assert isinstance(self.mega_facade.facade_layers, i12.FeatureMap)

        uncertainty = self.mega_facade.facade_layers.confidence[i12.WINDOW]
        uncertainty = uncertainty[self.sky_line:self.door_line, self.facade_left:self.facade_right]
        uncertainty = uncertainty.mean()

        return float(uncertainty)

    def find_regions(self, features):

        facade_strip = features[:, :, :, self.facade_left:self.facade_right]
        self.regions = {'window': extract_boxes_as_dictionaries(
            pyfacades.models.independant_12_layers.model.window(facade_strip)),
                        'door': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.door(facade_strip)),
                        'balcony': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.balcony(facade_strip)),
                        'sill': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.sill(facade_strip)),
                        'cornice': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.cornice(facade_strip)),
                        'shop': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.shop(facade_strip)),
                        'molding': extract_boxes_as_dictionaries(
                            pyfacades.models.independant_12_layers.model.molding(facade_strip))}

        # # Shift regions
        # for r in self.regions.values():
        #     for b in r:
        #         b['left'] += self.facade_left
        #         b['right'] += self.facade_left

    def to_dict(self):
        meta = dict()
        meta['uncertainty'] = self.uncertainty_for_windows()
        meta['facade-left'] = int(self.facade_left)
        meta['facade-right'] = int(self.facade_right)
        meta['sky-line'] = int(self.sky_line)
        meta['door-line'] = int(self.door_line)
        meta['win-count'] = int(self.win_count)
        meta['window-grid'] = self.window_grid.to_dict()
        meta['win-v-margin'] = make_list(self.window_grid.vertical_scores)
        meta['win-h-margin'] = make_list(self.window_grid.horizontal_scores)
        meta['rgb'] = make_list(self.color)
        meta['mezzanine'] = dict()
        meta['mezzanine']['rgb'] = make_list(self.mezzanine_color)
        meta['regions'] = self.regions
        return meta

    def _plot_background(self, bgimage):
        import pylab as pl
        # Show the portion of the image behind this facade
        left, right = self.facade_left, self.facade_right
        top, bottom = 0, self.mega_facade.rectified.shape[0]
        if bgimage is not None:
            pl.imshow(bgimage[top:bottom, left:right], extent=(left, right, bottom, top))
        else:
            # Fit the facade in the plot
            y0, y1 = pl.ylim()
            x0, x1 = pl.xlim()
            x0 = min(x0, left)
            x1 = max(x1, right)
            y0 = min(y0, top)
            y1 = max(y1, bottom)
            pl.xlim(x0, x1)
            pl.ylim(y1, y0)

    @property
    def width(self):
        return self.facade_right - self.facade_left

    def plot(self, bgimage=None):
        import pylab as pl

        self._plot_background(bgimage)
        ax = pl.gca()
        y0, y1 = pl.ylim()
        # r is the width of the thick line we use to show the facade colors
        r = 5
        patch = pl.Rectangle((self.facade_left + r, self.sky_line + r),
                             self.width - 2 * r,
                             self.door_line - self.sky_line - 2 * r,
                             color=self.color, fill=False, lw=2 * r)
        ax.add_patch(patch)

        pl.text((self.facade_right + self.facade_left) / 2.,
                (self.door_line + self.sky_line) / 2.,
                '$\sigma^2={:0.2f}$'.format(self.uncertainty_for_windows()))

        patch = pl.Rectangle((self.facade_left + r, self.door_line + r),
                             self.width - 2 * r,
                             y0 - self.door_line - 2 * r,
                             color=self.mezzanine_color, fill=False, lw=2 * r)
        ax.add_patch(patch)

        # Plot the left and right edges in yellow
        pl.vlines([self.facade_left, self.facade_right], self.sky_line, y0, colors='yellow')

        # Plot the door line and the roof line
        pl.hlines([self.door_line, self.sky_line], self.facade_left, self.facade_right, linestyles='dashed',
                  colors='yellow')

        self.window_grid.plot()

    def plot_regions(self, fill=True, bgimage=None, alpha=0.5):
        import pylab as pl
        ax = pl.gca()
        assert isinstance(ax, pl.Axes)

        colors = i12.JET_12

        self._plot_background(bgimage)

        for label in self.regions:
            color = colors[i12.LABELS.index(label)] / 255.

            for region in self.regions[label]:
                t = region['top']
                l = self.facade_left + region['left']
                b = region['bottom']
                r = self.facade_left + region['right']
                patch = pl.Rectangle((l, t), r - l, b - t, color=color, fill=fill, alpha=alpha)
                ax.add_patch(patch)


class MegaFacade(object):
    def __init__(self):
        super(MegaFacade, self).__init__()
        self.facade_merge_amount = 60  #about 1.5 meters
        self.wall_colors = None
        self.use_mask = True
        self.data = None
        self.path = ''
        self.rectified = None
        self.homography = None

        self.window_scores = None
        """I keep a modified version of the window scores so that I can mask some things out."""

        self._sky_mask = None
        self.facade_edge_scores = None
        self.data_mask = None
        self.driving_layers = None
        self.facade_layers = None
        self.facade_candidates = []
        # :type facade_candidates: list[FacadeCandidate]

    def _mask_out_common_obstructions(self):
        """Mask out the sky and some common objects that can obstruct a facade 
        
        This is intended to be run prior to rectifications, since these things can prevent us from 
        correctly identifying the rectilinear features of a facade
        
        """
        features = driving.process_strip(channels_first(img_as_ubyte(self.data)))

        if self.data_mask is not None:
            occlusions = ~self.data_mask
        else:
            occlusions = np.zeros(self.data.shape[:2], dtype=np.bool)
        occlusions |= driving.occlusion(features)
        self.data_mask = ~occlusions

    def _mask_out_wall_colors(self, facade_mask):
        wall_colors = self.rectified.copy()
        wall_colors[~facade_mask, :] = 0, 0, 0
        return wall_colors

    def _create_facade_mask(self):
        facade_mask = self.driving_layers.building() > 0.5
        facade_mask = binary_erosion(facade_mask, disk(10))  # Sky is noisy
        # Remove non-wall elements from the facade (we want just the wall)
        facade_mask[self.window_mask()] = 0
        facade_mask[self.facade_layers.door() > 0.5] = 0
        facade_mask[self.balcony_mask()] = 0
        # facade_mask[self.shop_mask()] = 0
        facade_mask[self.pillar_mask()] = 0
        facade_mask[self.facade_layers.molding() > 0.5] = 0
        return facade_mask

    def load_image(self, path):
        self.data = imread(path)
        self.path = path
        self._load_image_mask()
        self._mask_out_common_obstructions()
        self._rectify_image()

        self.driving_layers = driving.process_strip(channels_first(self.rectified * 255))
        self.facade_layers = i12.process_strip(channels_first(self.rectified * 255))

        self._create_sky_mask()
        self._segment_windows()
        self._segment_facade_edges()

        facade_cuts = self._split_at_facade_edges()
        facade_mask = self._create_facade_mask()
        wall_colors = self._mask_out_wall_colors(facade_mask)
        self.wall_colors = wall_colors

        self.facade_candidates = self._find_facade_candidates(wall_colors, facade_cuts)

    def sky_mask(self):
        return self._sky_mask

    def pillar_mask(self):
        return self.facade_layers.pillar() > 0.5

    def shop_mask(self):
        return self.facade_layers.shop() > 0.5

    def balcony_mask(self):
        return self.facade_layers.balcony() > 0.5

    def window_mask(self):
        return self.window_scores > 0.5

    def _split_at_facade_edges(self):
        facade_sig = self.facade_edge_scores.sum(0)
        facade_cuts = find_facade_cuts(facade_sig)
        return facade_cuts

    def _segment_facade_edges(self):
        # For facades, ignore the sky but do not use the mask (facade edges may be masked out)
        self.facade_edge_scores = self.facade_layers.facade_edge().copy()
        self.facade_edge_scores[self._sky_mask] = 0
        self.facade_edge_scores[self.facade_layers.door() > 0.5] = 0
        self.facade_edge_scores[self.window_mask()] = 0

    def door_mask(self):
        return i12.door(self.facade_layers) > 0.5

    def _segment_windows(self):
        # Ignore anything that seems to be sky
        self.window_scores = pyfacades.models.independant_12_layers.model.window(self.facade_layers).copy()
        if self.data_mask is not None:
            self.window_scores[~self.data_mask] = 0
        self.window_scores[self._sky_mask] = 0
        # Ignore anything that seems to be NOT part of a building
        self.window_scores[driving.occlusion(self.driving_layers)] = 0

    def _create_sky_mask(self):
        # Sky segmentation is poor / noisy, so use some morph ops to 'fix' it hackishly
        self._sky_mask = pyfacades.models.driving_12x360x480.model.sky(self.driving_layers) > 0.5
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
        if self.use_mask and os.path.isfile(mask_path):
            self.data_mask = rgb2gray(imread(mask_path)) > 0.5
        else:
            self.data_mask = None

    def _find_facade_candidates(self, wall_colors, cuts):
        facade_candidates = []
        for left, right in zip(cuts[:-1], cuts[1:]):
            facade_candidate = self._create_mini_facade(left, right, wall_colors)
            facade_candidates.append(facade_candidate)
        return facade_candidates

    def _create_mini_facade(self, left, right, wall_colors):
        door_strip = i12.door(self.facade_layers)[:, left:right].copy()
        shop_strip = i12.shop(self.facade_layers)[:, left:right]
        door_strip = np.max((door_strip, shop_strip), axis=0)
        win_strip = self.window_scores[:, left:right].copy()
        sky_strip = self._sky_mask[:, left:right].copy()
        rgb_strip = wall_colors[:, left:right]
        win_strip[:, :1] = win_strip[:, -1:] = 0  # edge effects
        sky_strip[:, :1] = sky_strip[:, -1:] = 0  # edge effects

        facade = FacadeCandidate(self, left, right, sky_strip, door_strip, win_strip, rgb_strip)
        facade.find_regions(self.facade_layers)
        return facade

    def to_dict(self):
        meta = dict()
        meta['facades'] = [facade.to_dict() for facade in self.facade_candidates]
        meta['homography'] = dict(inv=self.homography.inv_H.tolist(), tfm=self.homography.H.tolist())
        return meta

    def save_params(self, path=None):
        if path is None:
            path = join(dirname(self.path), 'parameters.yml')

        with open(path, 'wb') as f:
            meta = self.to_dict()
            data = yaml.dump(meta)
            f.write(yaml.dump(meta))
            assert "!!" not in data, "I cannot insert python-specific types:"

    def plot_grids(self):
        import pylab as pl
        pl.imshow(self.rectified)
        for facade in self.facade_candidates:
            facade.plot()

    def plot_regions(self, fill=True, alpha=0.5):
        import pylab as pl
        pl.imshow(self.rectified)
        for facade in self.facade_candidates:
            assert isinstance(facade, FacadeCandidate)
            facade.plot_regions(fill=fill, alpha=alpha)

    def save_plots(self, folder):

        import pylab as pl

        pl.gcf().set_size_inches(15, 15)

        pl.clf()
        self.homography.plot_original()
        pl.savefig(join(folder, 'homography-original.jpg'))

        pl.clf()
        self.homography.plot_rectified()
        pl.savefig(join(folder, 'homography-rectified.jpg'))

        pl.clf()
        self.driving_layers.plot(overlay_alpha=0.7)
        pl.savefig(join(folder, 'segnet-driving.jpg'))

        pl.clf()
        self.facade_layers.plot(overlay_alpha=0.7)
        pl.savefig(join(folder, 'segnet-i12-facade.jpg'))

        pl.clf()
        self.plot_grids()
        pl.savefig(join(folder, 'grid.jpg'))

        pl.clf()
        self.plot_regions()
        pl.savefig(join(folder, 'regions.jpg'))

        pl.clf()
        pl.gcf().set_size_inches(6, 4)
        self.plot_facade_cuts()
        pl.savefig(join(folder, 'facade-cuts.jpg'), dpi=300)
        pl.savefig(join(folder, 'facade-cuts.svg'))

        imsave(join(folder, 'walls.png'), self.wall_colors)

    def plot_facade_cuts(self):

        facade_sig = self.facade_edge_scores.sum(0)
        facade_cuts = find_facade_cuts(facade_sig, dilation_amount=self.facade_merge_amount)
        mu = np.mean(facade_sig)
        sigma = np.std(facade_sig)

        w = self.rectified.shape[1]
        pad=10

        gs1 = pl.GridSpec(5, 5)
        gs1.update(wspace=0.5, hspace=0.0)  # set the spacing between axes.

        pl.subplot(gs1[:3, :])
        pl.imshow(self.rectified)
        pl.vlines(facade_cuts, *pl.ylim(), lw=2, color='black')
        pl.axis('off')
        pl.xlim(-pad, w+pad)

        pl.subplot(gs1[3:, :], sharex=pl.gca())
        pl.fill_between(np.arange(w), 0, facade_sig, lw=0, color='red')
        pl.fill_between(np.arange(w), 0, np.clip(facade_sig, 0, mu+sigma), color='blue')
        pl.plot(np.arange(w), facade_sig, color='blue')

        pl.vlines(facade_cuts, facade_sig[facade_cuts], pl.xlim()[1], lw=2, color='black')
        pl.scatter(facade_cuts, facade_sig[facade_cuts])

        pl.axis('off')

        pl.hlines(mu, 0, w, linestyle='dashed', color='black')
        pl.text(0, mu, '$\mu$ ', ha='right')

        pl.hlines(mu + sigma, 0, w, linestyle='dashed', color='gray',)
        pl.text(0, mu + sigma, '$\mu+\sigma$ ', ha='right')
        pl.xlim(-pad, w+pad)



def main(argv=None):
    import argparse
    from glob import glob
    # noinspection PyShadowingNames
    import pyfacades.models.independant_12_layers as i12
    import caffe
    from os.path import join, basename, splitext
    from time import time

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('patterns', nargs='*')
    p.add_argument('--files',
                   help="A text file with the relative paths of images to process",
                   default='')
    p.add_argument('--weights',
                   help="The path to the trained weights to use",
                   default=i12.WEIGHTS)
    p.add_argument('--output', '-o',
                   help="The output folder. Will be created if it does not exist. ",
                   default=join(os.getcwd(), 'facades'))

    p.add_argument('--plot',
                   action='store_true',
                   help="Generate various diagnostic plots")
    p.add_argument('--no-use-mask',
                   dest='use_mask',
                   action="store_false",
                   help="Do not use the masks, even if provided. ")
    p.add_argument('--resume',
                   action='store_true',
                   help="Skip past previously generated output")

    args = p.parse_args(args=argv)

    # Process files based on args.pattern
    files = [f for pattern in args.patterns for f in glob(pattern)]
    outputs = [join(args.output, splitext(basename(f))[0], 'parameters.yml') for f in files]

    if args.files is not None and isfile(args.files):
        with open(args.files) as f:
            lines = f.readlines()
            lines = [line for line in lines if not line.strip()[0] == '#']
            col1, col2 = zip(*[line.split() for line in lines])
        more_files = [abspath(line.strip()) for line in col1]
        more_outputs = [join(args.output, dirname(filename), 'parameters.yml')
                        for filename in col2]
        files += more_files
        outputs += more_outputs

    # Set up the net / GPU
    caffe.set_mode_gpu()
    caffe.set_device(0)
    i12.net(args.weights)
    driving.net()

    n = len(files)
    for i in range(n):
        print i + 1, "of", n,

        if args.resume:
            if isfile(outputs[i]):
                print "skipping (resume=true)"
                continue

        start_time = time()
        mf = MegaFacade()
        mf.use_mask = args.use_mask
        assert isinstance(mf, MegaFacade)
        mf.load_image(files[i])

        try:
            os.makedirs(dirname(outputs[i]))
        except OSError:
            pass

        mf.save_params(outputs[i])

        imsave(join(dirname(outputs[i]), 'orthographic.png'), mf.data, )
        imsave(join(dirname(outputs[i]), 'rectified.png'), mf.rectified, )
        tend1 = time()

        if args.plot:
            import pylab
            pylab.close('all')
            mf.save_plots(dirname(outputs[i]))

        tend2 = time()

        total_time = tend2 - start_time
        output_time = tend1 - start_time
        plotting_time = tend2 - tend1
        print "--", basename(files[i]), "({:0.2f}s, {:0.2f}+{:0.2f})".format(total_time, output_time, plotting_time)




def ransac_guess_color(colors, n_iter=50, std=2):
    colors = rgb2lab(colors)
    colors = colors.reshape(-1, 3)
    masked = colors[:, 0] < 0.1
    colors = colors[~masked]
    assert len(colors) > 0, "Must have at least one color"

    best_mu = np.array([0, 0, 0])
    best_n = 0
    for k in range(n_iter):
        subset = colors[np.random.choice(np.arange(len(colors)), 1)]

        mu = subset.mean(0)
        #inliers = (((colors - mu) ** 2 / std) < 1).all(1)
        inliers = ((np.sqrt(np.sum((colors - mu)**2, axis=1))  / std) < 1)

        mu = colors[inliers].mean(0)
        n = len(colors[inliers])
        if n > best_n:
            best_n = n
            best_mu = mu
    #import ipdb; ipdb.set_trace()
    best_mu = np.squeeze(lab2rgb(np.array([[best_mu]])))
    return best_mu


def find_facade_cuts(facade_sig, dilation_amount=60):
    edges = facade_sig > facade_sig.mean() + facade_sig.std()
    edges = binary_closing(edges, structure=np.ones(dilation_amount))
    run, start, val = run_length_encode(edges)
    result = []
    for s, e in zip(start[val], start[val] + run[val]):
        result.append(s + facade_sig[s:e].argmax())
    result = [0] + result + [len(facade_sig) - 1]
    result = np.unique(result)
    return np.array(result)


def run_length_encode(input_array):
    """ run length encoding"""
    ia = np.array(input_array)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def extract_boxes_as_dictionaries(image, threshold=0.5, se=disk(3)):
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

if __name__ == '__main__':
    main()