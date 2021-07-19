import os
import random
import typing
import itertools
import numpy as np
from keras.utils import Sequence, to_categorical
from ..common import image_aug
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from PIL import Image
from deep_insight_face.common.image_aug import augment_img
from deep_insight_face.common.utils import img_read_n_resize, add_extension, read_pairs, InvalidPairsError


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def triplet_image_pairs(img_dir_path: str, pairs: str):
    path_pairs = []
    nb_classes = set()
    nrof_skipped_pairs = 0
    for pair in pairs:
        anchor, positive, negative = None, None, None
        try:
            if len(pair) == 4:
                anchor = add_extension(
                    os.path.join(img_dir_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                negative = add_extension(
                    os.path.join(img_dir_path, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                # Fetch Positive img
                _x = np.array([f for f in os.listdir(os.path.join(img_dir_path, pair[0])) if not f.startswith('.')])
                np.random.shuffle(_x)
                for pos in _x:
                    if pos != os.path.basename(anchor):
                        positive = os.path.join(img_dir_path, pair[0], pos)
                        break
        except InvalidPairsError:
            nrof_skipped_pairs += 1
        else:
            _flags = [True for x in [anchor, positive, negative] if x is not None and os.path.exists(x)]
            if len(_flags) == 3 and all(_flags):
                path_pairs.append((anchor, positive, negative))
                nb_classes.add(pair[0])
                nb_classes.add(pair[2])

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_pairs, list(nb_classes)


def facematch_image_pairs(img_dir_path, pairs):
    path_pairs = []
    nb_classes = set()
    nrof_skipped_pairs = 0
    for pair in pairs:
        path0, path1, issame = None, None, False
        try:
            if len(pair) == 3:
                path0 = add_extension(
                    os.path.join(img_dir_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(
                    os.path.join(img_dir_path, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = add_extension(
                    os.path.join(img_dir_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(
                    os.path.join(img_dir_path, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
        except InvalidPairsError:
            nrof_skipped_pairs += 1
        else:
            _flags = [True for x in [path0, path1] if x and os.path.exists(x)]
            if len(_flags) == 2 and all(_flags):    # Only add the pair if both paths exist
                path_pairs.append((path0, path1, issame))
                nb_classes.add(pair[0])
                nb_classes.add(pair[2])

    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_pairs, list(nb_classes)


def create_pairs(img_dir_path, func=None, pairs_txt='pairs.txt'):
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    assert func is not None, "func should be of type Callable"
    # Convert test image to bin for evaluation
    pairs = read_pairs(os.path.expanduser(pairs_txt))
    pairs, nb_classes = func(img_dir_path, pairs)
    cls_indices = sorted(nb_classes)
    name_to_idx = list(map(lambda cl: cls_indices.index(cl), nb_classes))
    on_hot = to_categorical(name_to_idx, num_classes=len(nb_classes))
    return pairs, nb_classes, on_hot


def triplet_datagenerator(
    img_dir_path: str,
    pairs_txt: str,
    rescale: float = 1. / 255.,
    do_augment: bool = True,
    batch_size: int = 64,
    target_size: typing.Tuple[int] = (160, 160),
    n_channels: int = 3, gray: bool = False
) -> typing.Iterator:

    img_pairs, nb_classes, on_hot = create_pairs(
        img_dir_path, func=facematch_image_pairs, pairs_txt=pairs_txt
    )

    pairs = list(zip(img_pairs, nb_classes, on_hot))
    random.shuffle(pairs)
    zipped = itertools.cycle(pairs)

    def _read_img(p: str):
        img = img_read_n_resize(p, target_size, rescale=1 / 255.0)
        if gray:
            img = image_aug.rgb_to_grayscale(img, channels=n_channels)
        return img

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img_pairs, nb_class, on_hot = next(zipped)

            # For triplet extract 3 pairs
            im1, im2, im3 = (np.array(_read_img(img), dtype=np.uint8) for img in img_pairs)

            if do_augment:
                im1, im2, im3 = [augment_img(im, augmentation_name='non_geometric') for im in [im1, im2, im3]]

            X.append(np.array([im1, im2, im3], dtype=np.uint8))
            Y.append(on_hot)

        X, Y = preprocess_input(np.array(X)), np.array(Y)

        yield X, Y


def facematch_datagenerator(
    img_dir_path: str,
    pairs_txt: str,
    rescale: float = 1. / 255.,
    do_augment: bool = True,
    batch_size: int = 64,
    target_size: typing.Tuple[int] = (160, 160),
    n_channels: int = 3, gray: bool = False
) -> typing.Iterator:

    img_pairs, nb_classes, on_hot = create_pairs(
        img_dir_path, func=facematch_image_pairs, pairs_txt=pairs_txt
    )

    pairs = list(zip(img_pairs, nb_classes, on_hot))
    random.shuffle(pairs)
    zipped = itertools.cycle(pairs)

    def _read_img(p: str):
        img = img_read_n_resize(p, target_size, rescale=1.0)
        if gray:
            img = image_aug.rgb_to_grayscale(img, channels=n_channels)
        return img

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            img_pairs, nb_class, on_hot = next(zipped)

            # For facematch/siamese extract 2 pairs
            im1, im2 = (np.array(_read_img(img), dtype=np.uint8) for img in img_pairs)

            if do_augment:
                im1, im2 = [augment_img(im) for im in [im1, im2]]

            X.append(np.array([im1, im2], dtype=np.uint8))
            Y.append(on_hot)

        X, Y = preprocess_input(np.array(X)), np.array(Y)

        yield X, Y


def get_train_dataset(pairs_txt: str, img_dir_path: str, generator_fn: iter,
                      img_shape=(160, 160, 3), is_train=True, batch_size=64):
    """
    Function to convert Keras Sequence to tensorflow dataset

    Arguments:
        pairs_txt {[type]} -- [description]
        img_dir_path {[type]} -- [description]
        generator_fn {[type]} -- [description]

    Keyword Arguments:
        img_shape {[type]} -- [description] (default: {(160, 160, 3)})
        is_train {[type]} -- [description] (default: {True})
        batch_size {[type]} -- [description] (default: {64})

    Returns:
        [type] -- [description]
    """
    image_gen = generator_fn(
        pairs_txt,
        img_dir_path,
        batch_size=batch_size,
        horizontal_rotation=True,
        preprocess_func=preprocess_input,
    )

    classes = image_gen.nb_classes
    steps_per_epoch = np.floor(len(image_gen.pairs) / batch_size)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(
        image_gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None, 3, *img_shape], [None, 1])
    )

    if is_train:
        train_ds = train_ds.repeat()

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, steps_per_epoch, classes
