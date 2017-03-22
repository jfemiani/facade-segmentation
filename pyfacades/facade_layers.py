import caffe
from glob import glob
import numpy as np
import os
import random
import skimage.transform

# noinspection PyPep8Naming
from pyfacades.rectify import H as homography


# noinspection PyPep8Naming
def random_perspective_transform(data, amt=20):
    d_left, d_right, d_top, d_bottom = np.random.randint(-amt, amt, 4)
    M = homography(d_left, d_right, d_top, d_bottom, 512, 512)
    data[...] = skimage.transform.warp(data.transpose(1, 2, 0),
                                       skimage.transform.ProjectiveTransform(M),
                                       order=0,
                                       preserve_range=True,
                                       cval=1,
                                       ).transpose(2, 0, 1)
    return data


class InputLayer(caffe.Layer):
    def __init__(self, p_object, *args, **kwargs):
        super(InputLayer, self).__init__(p_object, *args, **kwargs)
        self.num_channels = 3
        self.num_labels = 12
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.epochs = 0
        self.counter = 0
        self.source = 'output_stacks2'
        self.files = glob(os.path.join(self.source, '*.npy'))

    # noinspection PyMethodOverriding
    def setup(self, bottom, top):
        random.shuffle(self.files)

        print 'source:', self.source
        print 'number of files:', len(self.files)

    # noinspection PyMethodOverriding
    def reshape(self, bottom, top):
        # no "bottom"s for input layer
        if len(bottom) > 0:
            raise Exception('cannot have bottoms for input layer')

        # make sure you have the right number of "top"s
        assert len(top) == self.num_labels  # +1 for RGB,  -1 for the 'background' label.

        top[0].reshape(self.batch_size, self.num_channels, self.height, self.width)

        # Label 0 is background, we can / should safely ignore it!
        for i in range(1, self.num_labels):
            top[i].reshape(self.batch_size, 1, self.width, self.height)

    # noinspection PyUnusedLocal
    def forward(self, bottom, top):
        # do your magic here... feed **one** batch to `top`

        for i in range(self.batch_size):
            current = np.load(self.files[self.counter])

            self._transform(current)

            data = current[:3]
            labels = current[3:]
            top[0].data[i, ...] = data
            for j in range(1, len(labels)):
                top[j].data[i, ...] = labels[j]
            self.counter += 1

        # Reshuffle the images at the end of each epoch
        if self.counter >= len(self.files):
            random.shuffle(self.files)
            self.counter = 0
            self.epochs += 1

    @staticmethod
    def _transform(data):
        random_perspective_transform(data)

    def backward(self, top, propagate_down, bottom):
        # no back-prop for input layers
        pass
