import os
import tensorflow as tf
from typing import Tuple
import pickle


def tf_imread(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)  # [0, 255]
    # img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
    img = tf.cast(img, "float32")  # [0, 255]
    return img


class raw_image_to_tf:
    """ Convert RAW jpeg image to binary compressed file using tensorflow
    """

    def __init__(self, image_path: str, pairs_txt: str, test_bin_file: str,
                 image_size: Tuple[int] = (112, 112, 3)) -> None:
        self.image_path = image_path
        self.pairs_txt = pairs_txt
        self.test_bin_file = test_bin_file
        self.image_size = image_size

    def __add_extension__(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __get_paths__(self, lfw_dir, pairs):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            path0, path1, issame = None, None, False
            try:
                if len(pair) == 3:
                    path0 = self.__add_extension__(os.path.join(
                        lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = self.__add_extension__(os.path.join(
                        lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                    issame = True
                elif len(pair) == 4:
                    path0 = self.__add_extension__(os.path.join(
                        lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = self.__add_extension__(os.path.join(
                        lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                    issame = False
            except RuntimeError:
                nrof_skipped_pairs += 1
            else:
                if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                    path_list += (path0, path1)
                    issame_list.append(issame)
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list

    def __read_pairs__(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split('\t')
                pairs.append(pair)
        return pairs

    def _encode_img(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.encode_jpeg(tf.image.decode_jpeg(img, channels=3))
        return img

    def __call__(self) -> None:
        # Convert test image to bin for evaluation
        pairs = self.__read_pairs__(os.path.expanduser(self.pairs_txt))
        paths, issame_list = self.__get_paths__(self.image_path, pairs)
        # paths = list(itertools.chain(*paths))

        bb = list(map(self._encode_img, paths))
        print("Saving to %s" % self.test_bin_file)
        with open(self.test_bin_file, "wb") as ff:
            pickle.dump([bb, issame_list], ff)


def _cli():
    argv = parse_arguments()
    raw_image_to_tf(argv.image_path, argv.pairs_txt, argv.test_bin_file)()
    print(">>> Completed conversion .... >>>>")


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser("Method to verify aargument parser")
    parser.add_argument("--test_bin_file", default=None, type=str, help="Test bin file if available")
    parser.add_argument("--image_path", default=None, type=str, help="Image path to eval")
    parser.add_argument("--pairs_txt", default=None, type=str, help="Pairs file")
    return parser.parse_args()


if __name__ == "__main__":
    _cli()
