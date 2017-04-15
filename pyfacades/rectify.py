# Requires:
#  conda install scipy scikit-image scikit-learn
import skimage
import skimage.transform as tf
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from skimage.color import rgb2grey
import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model

RANSAC_OPTIONS = dict(residual_threshold=1)
OPTIMIZATION_METHOD = 'Nelder-Mead'
OPTIMIZATION_OPTIONS = dict(xatol=.0000001)


def prj(x):
    """Project by dividing through by the last coordinate.
    :param x: A point specified using projective coordinates
    :type x: array
    """
    return x[:2] / x[2]


def H_v(left, right, width, length):
    """Vertical factor of a projection matrix.

    :param left:
    :param right:
    """
    width = width + 0.  # Force floating point precision
    length = length + 0.
    return np.array([[1 + (right - left) / width, -left / length, left],
                     [0, 1 + (right - left) / width, 0],
                     [0, (right - left) / (width * length), 1]])


def H_h(top, bottom, width, length):
    """Vertical factor of a projection matrix.

     :param top:
     :param bottom:
     """
    width = width + 0.  # Force floating point precision
    length = length + 0.
    return np.array([[1 + (bottom - top) / length, 0, 0],
                     [-top / width, 1 + (bottom - top) / length, top],
                     [(bottom - top) / (width * length), 0, 1]])


def H(dleft, dright, dtop, dbottom, width, length):
    return H_v(dleft, dright, width, length).dot(H_h(dtop, dbottom, width, length))


from pylab import *

__i__ = 0
def _extract_lines(img, edges=None, mask=None, min_line_length=20, max_line_gap=3):
    global __i__
    __i__ += 1

    if edges is None:
        edges = canny(rgb2grey(img))
    if mask is not None:
        edges = edges & mask

    figure()
    subplot(131)
    imshow(img)
    subplot(132)
    imshow(edges)
    subplot(133)
    imshow(mask, cmap=cm.gray)
    savefig('/home/shared/Projects/Facades/src/data/for-labelme/debug/foo/{:06}.jpg'.format(__i__))

    lines = np.array(probabilistic_hough_line(edges, line_length=min_line_length, line_gap=max_line_gap))

    return lines


def _vlines(lines, ctrs=None, lengths=None, vecs=None, angle_lo=20, angle_hi=160, ransac_options=RANSAC_OPTIONS):
    ctrs = ctrs if ctrs is not None else lines.mean(1)
    vecs = vecs if vecs is not None else lines[:, 1, :] - lines[:, 0, :]
    lengths = lengths if lengths is not None else np.hypot(vecs[:, 0], vecs[:, 1])

    angles = np.degrees(np.arccos(vecs[:, 0] / lengths))
    points = np.column_stack([ctrs[:, 0], angles])
    point_indices, = np.nonzero((angles > angle_lo) & (angles < angle_hi))
    points = points[point_indices]
    if len(points) > 0:
        model_ransac = linear_model.RANSACRegressor(**ransac_options)
        model_ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1))
        inlier_mask = model_ransac.inlier_mask_
        valid_lines = lines[point_indices[inlier_mask], :, :]
    else:
        valid_lines = []
    return valid_lines


def _hlines(lines, ctrs=None, lengths=None, vecs=None, angle_lo=20, angle_hi=160, ransac_options=RANSAC_OPTIONS):
    ctrs = ctrs if ctrs is not None else lines.mean(1)
    vecs = vecs if vecs is not None else lines[:, 1, :] - lines[:, 0, :]
    lengths = lengths if lengths is not None else np.hypot(vecs[:, 0], vecs[:, 1])

    angles = np.degrees(np.arccos(vecs[:, 1] / lengths))
    points = np.column_stack([ctrs[:, 1], angles])
    point_indices, = np.nonzero((angles > angle_lo) & (angles < angle_hi))
    points = points[point_indices]
    if len(points) == 0:
        model_ransac = linear_model.RANSACRegressor(**ransac_options)
        model_ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1))
        inlier_mask = model_ransac.inlier_mask_
        valid_lines = lines[point_indices[inlier_mask], :, :]
    else:
        valid_lines = []
    return valid_lines


def _vh_lines(lines, ctrs=None, lengths=None, vecs=None, angle_lo=20, angle_hi=160,
              ransac_options=RANSAC_OPTIONS):
    assert len(lines) > 0, "We need some lines to start with!"
    ctrs = ctrs if ctrs is not None else lines.mean(1)
    vecs = vecs if vecs is not None else lines[:, 1, :] - lines[:, 0, :]
    lengths = lengths if lengths is not None else np.hypot(vecs[:, 0], vecs[:, 1])

    return (_vlines(lines, ctrs, lengths, vecs, angle_lo, angle_hi, ransac_options=RANSAC_OPTIONS),
            _hlines(lines, ctrs, lengths, vecs, angle_lo, angle_hi, ransac_options=RANSAC_OPTIONS))


def _solve_lr(vlines, w, l, opt_options=OPTIMIZATION_OPTIONS, opt_method=OPTIMIZATION_METHOD, limit=0.3):
    """ Solve for the left and right edge displacement.
    This routine estimates the amount to move the upper left and right cornders of the image
     in a horizontal direction in order to make the given lines parallel and vertical.

    :param vlines: Lines that we want to map to vertical lines.
    :param w:  The width of the image
    :param l:   The height of the image
    :param opt_options: Optimization options passed into `minimize`
    :param opt_method: The optimization method.
    :param limit:  A limit on the amount of displacement -- beyond this and we will assume failure.
    :return: (dl, dr),   the horizontal displacement of the left and right corners.
    """
    if len(vlines) == 0:
        return 0, 0

    a = np.append(vlines[:, 0, :], np.ones((len(vlines), 1)), axis=1)
    b = np.append(vlines[:, 1, :], np.ones((len(vlines), 1)), axis=1)

    def objective(x):
        dl, dr = x
        Hv = np.linalg.inv(H_v(dl, dr, w, l))
        return np.sum(np.abs(Hv[0, :].dot(a.T) / Hv[2, :].dot(a.T) - Hv[0, :].dot(b.T) / Hv[2, :].dot(b.T)))

    res = minimize(objective, (0., 0.),
                   options=opt_options,
                   method=opt_method)
    dl, dr = res.x

    # Give up if the solution is not plausible (this indicates that the 'vlines' are too noisy
    if abs(dl) > limit * w:
        dl = 0
    if abs(dr) > limit * w:
        dr = 0
    return dl, dr


def _solve_ud(hlines, dl, dr, w, l, opt_options=OPTIMIZATION_OPTIONS, opt_method=OPTIMIZATION_METHOD, limit=0.3):
    """ Solve for the left top and bottom edge displacement.
    This routine estimates the amount to move the upper left and lower left corners of the image
    in a vertical direction in order to make the given lines parallel and horizontal.

    :param hlines: Lines that we want to map to horizontal lines.
    :param w:  The width of the image
    :param l:   The height of the image
    :param opt_options: Optimization options passed into `minimize`
    :param opt_method: The optimization method.
    :param limit:  A limit on the amount of displacement -- beyond this and we will assume failure.
                   It is expressed as a fraction of the image height.
    :return: (dl, dr),   the horizontal displacement of the left and right corners.
    """
    if len(hlines) == 0:
        return 0, 0

    a = np.append(hlines[:, 0, :], np.ones((len(hlines), 1)), axis=1)
    b = np.append(hlines[:, 1, :], np.ones((len(hlines), 1)), axis=1)

    Hv = np.linalg.inv(H_v(dl, dr, w, l))
    a = Hv.dot(a.T).T
    b = Hv.dot(b.T).T

    def objective(x):
        du, dd = x
        Hh = np.linalg.inv(H_h(du, dd, w, l))
        return np.sum(np.abs(Hh[1, :].dot(a.T) / Hh[2, :].dot(a.T) - Hh[1, :].dot(b.T) / Hh[2, :].dot(b.T)))

    res = minimize(objective, (0., 0.),
                   options=opt_options,
                   method=opt_method)
    du, dd = res.x

    # Give up if the result is not plausible. We are better off nor warping.
    if abs(du) > limit * l:
        dl = 0
    if abs(du) > limit * l:
        dr = 0
    return du, dd


def _solve_lrud(hlines, vlines, w, l, opt_options=OPTIMIZATION_OPTIONS, opt_method=OPTIMIZATION_METHOD):
    dl_, dr_ = _solve_lr(vlines, w, l, opt_options=opt_options, opt_method=opt_method)
    du_, dd_ = _solve_ud(hlines, dl_, dr_, w, l, opt_options=opt_options, opt_method=opt_method)
    return dl_, dr_, du_, dd_


def rectify(image, **kwargs):
    """ Rectify an image

    see the Homagraphy class documentation for details.
    """
    h = Homography(image, **kwargs)
    return h.rectified


class Homography(object):
    def __init__(self, img,
                 ransac_options=RANSAC_OPTIONS,
                 opt_method=OPTIMIZATION_METHOD,
                 opt_options=OPTIMIZATION_OPTIONS,
                 mask=None,
                 min_line_length=20,
                 max_line_gap=3,
                 angle_tolerance=30
                 ):
        self.angle_tolerance = angle_tolerance
        self.ransac_options = ransac_options
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.data = img
        self.mask = mask
        if mask is not None and np.all(mask == False):
            self.mask = None

        self.l, self.w = img.shape[:2]
        self.lines = _extract_lines(img,
                                    mask=self.mask,
                                    min_line_length=min_line_length,
                                    max_line_gap=max_line_gap)
        if len(self.lines) > 0:
            self.vlines, self.hlines = _vh_lines(self.lines,
                                                 ransac_options=self.ransac_options,
                                                 angle_lo=90 - self.angle_tolerance,
                                                 angle_hi=90 + self.angle_tolerance
                                                 )
            lrud = _solve_lrud(self.hlines, self.vlines, self.w, self.l,
                               opt_options=opt_options,
                               opt_method=opt_method)
            self.dl, self.dr, self.du, self.dd = lrud
            self.H = H(self.dl, self.dr, self.du, self.dd, self.w, self.l)
            self.inv_H = np.linalg.inv(self.H)
            self.rectified = tf.warp(img, self.H)
            if mask is not None:
                self.rectified_mask = tf.warp(mask, self.H)
            else:
                self.rectified_mask = None
        else:
            self.vlines = []
            self.hlines = []
            self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            self.inv_H = self.H
            self.rectified = img.copy()
            if self.mask is not None:
                self.rectified_mask = self.mask.copy()
            else:
                self.rectified_mask = None

    def inv_transform(self, x):
        return prj(self.inv_H.dot(np.append(x, 1)))

    def transform(self, x):
        return prj(self.H.dot(np.append(x, 1)))

    def plot_rectified(self):
        import pylab
        pylab.title('rectified')
        pylab.imshow(self.rectified)

        for line in self.vlines:
            p0, p1 = line
            p0 = self.inv_transform(p0)
            p1 = self.inv_transform(p1)
            pylab.plot((p0[0], p1[0]), (p0[1], p1[1]), c='green')

        for line in self.hlines:
            p0, p1 = line
            p0 = self.inv_transform(p0)
            p1 = self.inv_transform(p1)
            pylab.plot((p0[0], p1[0]), (p0[1], p1[1]), c='red')

        pylab.axis('image');
        pylab.grid(c='yellow', lw=1)
        pylab.plt.yticks(np.arange(0, self.l, 100.0));
        pylab.xlim(0, self.w)
        pylab.ylim(self.l, 0)

    def plot_original(self):
        import pylab
        pylab.title('original')
        pylab.imshow(self.data)

        for line in self.lines:
            p0, p1 = line
            pylab.plot((p0[0], p1[0]), (p0[1], p1[1]), c='blue', alpha=0.3)

        for line in self.vlines:
            p0, p1 = line
            pylab.plot((p0[0], p1[0]), (p0[1], p1[1]), c='green')

        for line in self.hlines:
            p0, p1 = line
            pylab.plot((p0[0], p1[0]), (p0[1], p1[1]), c='red')

        pylab.axis('image');
        pylab.grid(c='yellow', lw=1)
        pylab.plt.yticks(np.arange(0, self.l, 100.0));
        pylab.xlim(0, self.w)
        pylab.ylim(self.l, 0)
