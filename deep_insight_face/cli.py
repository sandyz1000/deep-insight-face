from collections import namedtuple
import logging
import os
import click
from time import time
from functools import wraps


_BASE_DIRNAME = os.path.abspath(os.path.dirname(__file__) + os.sep + os.pardir)
LOG_DIR = os.path.join(_BASE_DIRNAME, "logs")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
MODEL_DIR = os.path.join(_BASE_DIRNAME, "weights/facematch")


ImageDataPath = namedtuple("ImageDataPath", ("img_path", "pairs"))


def timing(f):
    @wraps(f)
    def _wrap_func(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        print(f"{f.__name__} took: {te - ts} sec")
        return result
    return _wrap_func


def print_training_info(**kwargs):
    # NOTE: Use log instead of print
    logger.info("===================================")
    logger.info(">>> Initializing training: >>>")
    logger.info("\n" +
                f"Saving output Model path: {kwargs['model_path']} \n" +
                f"Input Shape: {kwargs['input_shape']} \n" +
                f"Input image directory: {kwargs['data_path']} \n" +
                f"Optional Evaluation directory: {kwargs['eval_paths']}"
                )
    logger.info(f"Mode: {kwargs.get('mode', None)}" if 'mode' in kwargs else "")
    logger.info("===================================")


@click.command()
@click.option("--model_path", type=str, default=None, help="Siamese model path")
@click.option("--input_shape", type=tuple, default=(96, 96, 3), help="Input image shape of the N/W")
@click.option("--data_path", required=True, type=str, help="Face-KYC aligned dataset path")
@click.option("--eval_paths", default=None, type=str, help="Evaluation dataset path or bin")
@click.option("--batch_size", type=int, default=64, help="Batch size of the training dataset, default size is 64")
@click.option("--emd_size", type=int, default=128, help="Embedding size of the bottleneck layer")
@click.option("--epochs", default=20, type=int, help="EPOCHS size, default to 20")
@click.option("--mode", default='simple_triplet_nw', type=str, show_choices=[1, 2],
              help="Define Loss function for backend CHOICES is 'simple_triplet_nw', 'semihard_triplet_nw' ")
def train_triplets(model_path, input_shape, data_path, eval_paths, batch_size=64,
                   emd_size=128, epochs=20, mode=1, threshold=0.5):
    from .networks.triplet import model_choice
    from .training.triplet import Train

    model_path = MODEL_DIR + os.sep + f"triplet/tripletface_epochs-{epochs}_mode-{mode}.h5"
    print_training_info(model_path=model_path, epochs=epochs, data_path=data_path, eval_path=eval_paths)
    train_data_path = ImageDataPath(data_path, data_path + os.path.sep + "pairs.txt")

    mode = getattr(model_choice, f"choice_{mode}")
    val_data = (ImageDataPath(eval_paths, eval_paths + os.path.sep + "pairs.txt")
                if os.path.isdir(eval_paths) else eval_paths)

    initarg = dict(model_path=model_path, input_shape=input_shape, emd_size=emd_size, mode=mode)

    init_training = Train(**initarg)
    init_training._fit(train_data_path, val_data, epochs=epochs, batch_size=batch_size, threshold=threshold)


@click.command()
@click.option("--backbone", type=str, default='mobinet', help="Model backbone")
@click.option("--model_path", type=str, default=None, help="Model checkpoint fullpath")
@click.option("--data_path", required=True, type=str, help="Face-KYC aligned dataset path")
@click.option("--eval_paths", nargs='+', default=[], type=str, help="Evaluation dataset path")
@click.option("--batch_size", type=int, default=32, help="Batch size of the training dataset default size is 32")
@click.option("--emd_size", type=int, default=256, help="Embedding size of the bottleneck layer")
@click.option("--epochs", default=20, type=int, help="EPOCHS size, default to 20")
def train_multiface(backbone, model_path, data_path, eval_paths, batch_size=128, emd_size=256, epochs=20):
    from .training.multiface import multiface_train, deepinsight_train, deepinsight_train2

    assert all([os.path.splitext(e)[1] == 'bin' for e in eval_paths]), "Invalid model extenstion, prepare in bin format"

    multiface_train(data_path, model_path, eval_paths, log_dir=os.path.join(LOG_DIR, "insightface"))
    deepinsight_train(backbone, data_path, model_path, eval_paths, log_dir=os.path.join(LOG_DIR, "insightface"))
    # deepinsight_train2(backbone, data_path, model_path, eval_paths, log_dir=os.path.join(LOG_DIR, "insightface"))


@click.command()
@click.option("--model_path", type=str, default=None, help="Siamese model path")
@click.option("--input_shape", type=tuple, default=(112, 112, 3), help="Input image shape of the N/W")
@click.option("--data_path", required=True, type=str, help="Face-KYC aligned dataset path")
@click.option("--eval_paths", default=None, type=str, help="Evaluation dataset path or bin file")
@click.option("--batch_size", type=int, default=32, help="Batch size of the training dataset default size is 32")
@click.option("--emd_size", type=int, default=128, help="Embedding size of the bottleneck layer")
@click.option("--epochs", default=20, type=int, help="EPOCHS size, default to 20")
def train_siamese(model_path, input_shape, data_path, eval_paths, batch_size=32, emd_size=128, epochs=20):
    from .training.siamese import Train
    model_path = model_path + os.path.sep + f"siameseface_epochs-{epochs}_mode-default.h5"
    print_training_info(model_path=model_path, epochs=epochs, data_path=data_path, eval_path=eval_paths)
    train_data = ImageDataPath(data_path, data_path + os.path.sep + "pairs.txt")
    val_data = (ImageDataPath(eval_paths, eval_paths + os.path.sep + "pairs.txt")
                if os.path.isdir(eval_paths) else eval_paths)

    initarg = dict(model_path=model_path, input_shape=input_shape, emd_size=emd_size)

    init_training = Train(**initarg)
    init_training._fit(train_data, val_data, epochs=epochs, batch_size=batch_size, threshold=0.4)


@click.command()
@click.option('--images_path', required=True, help='Path to the data directory containing aligned face patches.')
@click.option('model_path', required=True, help='Could be either a directory containing the meta_file and ckpt_file' +
              'or a model protobuf (.pb) file or *.h5 file')
@click.option('model_name', default="triplet", show_choices=[],
              help="Model Loss in which it is trained on. It determine the type of model use for evaluation")
@click.option('batch_size', default=12, help='Number of images to process in a batch in the test set.')
@click.option('image_size', default=160, help='Image size (height, width) in pixels.')
@click.option('pairs', default="pairs.txt", help='The file containing the pairs to use for validation.')
@click.option('nrof_folds', default=10, help='Number of folds to use for cross validation. Mainly used for testing.')
@click.option('distance_metric', default=10, help='Distance metric  0:euclidian, 1:cosine similarity.')
@click.option('use_flipped_images', default=True,
              help='Concatenates embeddings for the image and its horizontally flipped counterpart.')
@click.option('subtract_mean', default=True, help='Subtract feature mean before calculating distance.')
@click.option('use_fixed_image_standardization', default=True, help='Performs fixed standardization of images.')
@click.option('save_output_detail', default=True, help='Flags to specify weather to save image-wise output to file')
@click.option('result_csv_name', default="result.csv", help='Saves threshold wise csv in on provided path')
def evaluate(
        images_path, model_path, model_name, batch_size=12, image_size=160, pairs='pairs.txt',
        nrof_folds=10, distance_metric=10, use_flipped_images=True, subtract_mean=True,
        use_fixed_image_standardization=True, save_output_detail=True, result_csv_name="result.csv"):
    from .evaluation.evals import evaluate as api_evaluate
    
    # TODO: Fix this method
    model = None
    api_evaluate(model, images_path, pairs, batch_size,
                 nrof_folds, distance_metric, subtract_mean,
                 use_flipped_images, use_fixed_image_standardization)


@click.command()
@click.option("--test_bin_file", default=None, type=str, help="Test bin file if available")
@click.option("--image_path", default=None, type=str, help="Image path to eval")
@click.option("--pairs_txt", default=None, type=str, help="Pairs file")
@timing
def img_tf_bin(test_bin_file, image_path, pairs_txt):
    from .raw_img_tf import raw_image_to_tf
    raw_image_to_tf(image_path, pairs_txt, test_bin_file)()
    logger.info(">>> Completed Conversion IMG_TF_BIN ....")


@click.command()
@click.option('--image_dir', type=str, required=True, help='Path to the image directory.')
@click.option('--pairs_file_name', type=str, default='pairs.txt', help='Filename of pairs.txt')
@click.option('--num_folds', type=int, required=True, help='Number of folds for k-fold cross validation.')
@click.option('--write_to_file', type=bool, default=True, help="Write output to file")
@click.option('--num_matches_mismatches', type=int, required=True, help='Number of matches/mismatches per fold.')
@timing
def generate_pairs(image_dir, pairs_file_name, num_folds, write_to_file, num_matches_mismatches):
    from .generate_pairs import generate_pairs
    generate_pairs(image_dir, num_folds, num_matches_mismatches, write_to_file, pairs_file_name)
    logger.info(">>> Completed Generating Image pairs ....")


@click.group()
def main():
    return 0


main.add_command(train_siamese, "train_siamese")
main.add_command(train_multiface, "train_deepinsight")
main.add_command(train_triplets, "train_triplet")
main.add_command(evaluate, "evaluate")
main.add_command(img_tf_bin, 'img_tf_bin')
main.add_command(generate_pairs, 'generate_pairs')
