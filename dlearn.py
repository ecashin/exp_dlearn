#! /usr/bin/env python2.7
"""Learn some high-level features for images like these:

http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
"""

from os.path import basename, join

import click
from logbook import Logger, StderrHandler
import pandas as pd
from sklearn.decomposition import dict_learning_online
from skimage.data import imread
from skimage.transform import resize


log = Logger(basename(__file__))


IMG_NROW = 200
IMG_NCOL = 220
N_ITER = 10
BATCH_SIZE = 1


def prep(image_name):
    bname = basename(image_name)
    log.info('reading {}'.format(bname))
    image = imread(image_name, as_grey=True)
    log.info('resizing {}'.format(bname))
    return resize(image, (IMG_NROW, IMG_NCOL))


@click.command()
@click.option(
    '--output',
    type=click.File('wb'),
    default='-')
@click.argument(
    'image_files',
    nargs=-1,
    type=click.Path(exists=True))
def main(output, image_files):
    log.info('starting with {} image files'.format(len(image_files)))
    images = (prep(i) for i in image_files)
    log.info('starting online dictionary learning')
    D = None
    for image in images:
        D = dict_learning_online(
            image,
            dict_init=D,
            n_components=2000,
            verbose=True,
            n_jobs=-1,
            n_iter=N_ITER,
            batch_size=BATCH_SIZE,
            return_code=False)
    output.write(pd.DataFrame(D).to_csv())
    log.info('done')


if __name__ == '__main__':
    with StderrHandler().applicationbound():
        main()
