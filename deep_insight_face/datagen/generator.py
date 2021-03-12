import os
import random
import itertools
import numpy as np
import threading
from keras.utils import Sequence, to_categorical
import tensorflow.python.keras.backend as K
from ..common import image_aug
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from PIL import Image
from ..common.image_aug import augment_img
from ..utils import img_read_n_resize, add_extension, read_pairs, InvalidPairsError


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


def triplet_image_pairs(img_dir_path, pairs):
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


class FaceMatchDataGenerator(Sequence):
    """
    Return an Instance of keras.utils.Sequence. 
    It's inherits keras.utils.Sequence which has all the goodies to iterate facial dataset
    """
    # ===========================================================================
    # BELOW CODE AN IMPLEMENTATION OF SIMPLE TRIPLET AND SIAMESE DATAGENERATOR
    # ===========================================================================

    def __init__(self,
                 image_paths,
                 pairs_txt,
                 rescale=1. / 255.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 preprocess_func=None,
                 batch_size=64,
                 target_size=(160, 160),
                 shuffle=True, n_channels=3, seed=42, gray=False):
        self.lock = threading.Lock()
        data_format = K.image_data_format()
        self.batch_size = batch_size
        self.rescale = rescale
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.nb_classes = -1
        self.target_size = target_size
        self.pairs = []
        self.indexes = []
        self.PAIR = 2
        self.gray = gray
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.preprocess_func = (lambda x, *args: preprocess_func(x, *args) if preprocess_func else x)
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.create_pairs(image_paths, pairs_txt)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, indices):
        raise NotImplementedError(
            f"Method {self.__data_generation__.name} should be implementated by child class")

    def create_pairs(self, img_dir_path, pairs_txt='pairs.txt'):
        raise NotImplementedError(
            f"Method {self.create_pairs.__name__} should be implementated by child class")

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation__(indexes)
        return X, y

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        # This also denotes default steps-per-epoch
        return int(np.floor(len(self.pairs) / self.batch_size))


class SiameseDataGenerator(FaceMatchDataGenerator):
    """
    Siamese Datagenerator that create a pairs of in a format of 
    (anchor, negative) -> 0 and (anchor, positive) -> 1
    """

    def create_pairs(self, img_dir_path, pairs_txt):
        '''
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        # Convert test image to bin for evaluation
        self.pairs, self.nb_classes, self.on_hot = create_pairs(
            img_dir_path, func=facematch_image_pairs, pairs_txt=pairs_txt
        )

    def __data_generation__(self, indices):
        X = np.zeros((self.batch_size, self.PAIR, *self.target_size, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(indices):
            pair = self.pairs[ID]
            img_pairs = [img_read_n_resize(p, self.target_size, self.rescale) for p in [pair[0], pair[1]]]
            if self.gray:
                img_pairs = [image_aug.rgb_to_grayscale(x) for x in img_pairs]
            if self.preprocess_func:
                img_pairs = [self.preprocess_func(x, 2) for x in img_pairs]

            if self.vertical_flip:
                img_pairs = [image_aug.flip_axis(x, self.col_axis - 1) for x in img_pairs]
            if self.horizontal_flip:
                img_pairs = [image_aug.flip_axis(x, self.row_axis - 1) for x in img_pairs]

            is_same = pair[2]
            X[i, ...] = np.array(img_pairs)
            y[i, ...] = is_same

        X = np.split(X, self.PAIR, axis=1)
        X = list(map(lambda item: np.squeeze(item), X))
        return X, y


class TripletGenerator(FaceMatchDataGenerator):
    """A Simple triplet Data generator for training triplets of faces/images
    with a predifined pairs of faces, an alternatives option to use batch hard triplet
    generator for optimized training on the negative sets
    """

    def __init__(self, *args, **kwargs):
        super(TripletGenerator, self).__init__(*args, **kwargs)
        self.PAIR = 3

    def __data_generation__(self, indices):
        X = np.zeros((self.batch_size, self.PAIR, *self.target_size, self.n_channels))
        y = np.zeros((self.batch_size), dtype=int)

        for i, ID in enumerate(indices):
            pair = self.pairs[ID]
            # NOTE: Always use RGB format while training
            img_pairs = [img_read_n_resize(p, self.target_size, self.rescale) for p in pair]
            if self.gray:
                img_pairs = [image_aug.rgb_to_grayscale(x) for x in img_pairs]
            if self.preprocess_func:
                img_pairs = [self.preprocess_func(x, 2) for x in img_pairs]

            if self.vertical_flip:
                img_pairs = [image_aug.flip_axis(x, self.col_axis - 1) for x in img_pairs]
            if self.horizontal_flip:
                img_pairs = [image_aug.flip_axis(x, self.row_axis - 1) for x in img_pairs]
            X[i, ...] = np.array(img_pairs)
            y[i, ...] = 1

        X = np.split(X, self.PAIR, axis=1)
        X = list(map(lambda item: np.squeeze(item), X))

        return X, y

    def create_pairs(self, img_dir_path, pairs_txt):
        '''
        Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        # Convert test image to bin for evaluation
        self.pairs, self.nb_classes, self.on_hot = create_pairs(
            img_dir_path, func=triplet_image_pairs, pairs_txt=pairs_txt
        )

    def _num_classes(self):
        return [i for i in self.pairs]


def triplet_datagenerator(img_dir_path,
                          pairs_txt,
                          rescale=1. / 255.,
                          do_augment=True,
                          batch_size=64,
                          target_size=(160, 160),
                          n_channels=3, gray=False):

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

            # For triplet extract 3 pairs
            im1, im2, im3 = (np.array(_read_img(img), dtype=np.uint8) for img in img_pairs)

            if do_augment:
                im1, im2, im3 = [augment_img(im) for im in [im1, im2, im3]]

            X.append(np.array([im1, im2, im3], dtype=np.uint8))
            Y.append(on_hot)

        X, Y = preprocess_input(np.array(X)), np.array(Y)
        
        yield X, Y


def facematch_datagenerator(img_dir_path,
                            pairs_txt,
                            rescale=1. / 255.,
                            do_augment=True,
                            batch_size=64,
                            target_size=(160, 160),
                            n_channels=3, gray=False):
    
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


def get_train_dataset(pairs_txt: str, img_dir_path: str, generator_fn: tf.keras.utils.Sequence,
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
