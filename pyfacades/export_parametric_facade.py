import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import binary_opening
from skimage.morphology import disk


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


