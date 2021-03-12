import os
import math
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate(embeddings, labels,
             nrof_folds=10,
             distance_metric=0,
             subtract_mean=False,
             thresholds=np.arange(0, 4, 0.01)):
    """ Evaluation Helper that will calculate metrics for all the embedding score
    """
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, f1scores = calculate_roc(thresholds, embeddings1, embeddings2,
                                                 np.asarray(labels),
                                                 nrof_folds=nrof_folds,
                                                 distance_metric=distance_metric,
                                                 subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    far_target = 1e-3
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(labels),
                                      far_target,
                                      nrof_folds=nrof_folds,
                                      distance_metric=distance_metric,
                                      subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, f1scores, val, val_std, far


def calculate_accuracy(threshold, dist, actual_issame, display_cm=False):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp)
    recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    f1score = 0 if float(precision + recall) == 0.0 else 2 * (float(precision * recall) / float(precision + recall))
    return tpr, fpr, acc, f1score


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise RuntimeError('Undefined distance metric %d' % distance_metric)

    return dist


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = 0 if n_same == 0 else float(true_accept) / float(n_same)
    far = 0 if n_diff == 0 else float(false_accept) / float(n_diff)
    return val, far


def calculate_val(thresholds, embeddings1, embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10,
                  distance_metric=0,
                  subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_roc(thresholds, embeddings1, embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  distance_metric=0,
                  subtract_mean=False):
    """
    Calculate ROC for the given thresholds
    """
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    f1scores = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1 - mean, embeddings2 - mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        f1score_train = np.zeros(shape=(nrof_thresholds))

        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], f1score_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])

        # Find best threshold based on F1 Score or Accuracy, uncomment if you want to evaluate on different metric
        best_threshold_index = np.argmax(acc_train)
        # best_threshold_index = np.argmax(f1score_train)

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], f1scores[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        print("Best Threshold value %02d" % best_threshold_index)
    return tpr, fpr, accuracy, f1scores


def get_emd_distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 0)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def print_confusion_matrix(results, threshold):
    tp = results[0][0]
    tn = results[1][1]
    fn = results[0][1]
    fp = results[1][0]
    coeff = 0.001
    # coeff = 1
    recall = tp / (tp + fn + coeff)
    precision = tp / (tp + fp + coeff)
    f1 = 2 * (precision * recall) / (precision + recall + coeff)

    print("TOTAL TP = {},TN = {},FP = {},FN ={}".format(tp, tn, fp, fn))

    confusion_matrix = \
        """
               | same   | different  | TRUTH
    ---------- | ------ | ---------- | -----
         same  | {:<5}  | {:<5}      |
    different  | {:<5}  | {:<5}      |
    PREDICTION |
    """
    print(confusion_matrix.format(tp, fp, fn, tn))

    # show_image_pairs(incorrect[:], "output", "FP-plot")
    print("Threshold: ", threshold)
    print("Accuracy: ", ((tp + tn) / (tp + tn + fp + fn + coeff)) * 100)
    print("recall: ", recall)
    print("precision: ", precision)
    print("F1 Score: ", f1)


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        path0, path1, issame = None, None, False
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split('\t')
            pairs.append(pair)
    return np.array(pairs)


class IdentityMetadata(object):
    def __init__(self, file_path):
        # image file name
        assert os.path.exists(file_path)
        self.file_path = file_path
        self.name = os.path.splitext(self.file_path.split("/")[-1])[0]

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return self.file_path


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                file_path = os.path.join(path, i, f)
                metadata.append(IdentityMetadata(file_path))
    return np.array(metadata)


def load_img_pair(img1, img2):
    metadata = []
    metadata.append(IdentityMetadata(img1))
    metadata.append(IdentityMetadata(img2))
    return np.array(metadata)


def rename(person_folder):
    """Renames all the images in a folder in lfw format

    Arguments:
        person_folder {str} -- path to folder named after person
    """
    all_image_paths = glob.glob(os.path.join(person_folder, "*.*"))
    all_image_paths = sorted([image for image in all_image_paths if image.endswith(
        ".jpg") or image.endswith(".png")])
    person_name = os.path.basename(os.path.normpath(person_folder))
    concat_name = '_'.join(person_name.split())
    for index, image_path in enumerate(all_image_paths):
        image_name = concat_name + '_' + '%04d' % (index + 1)
        file_ext = Path(image_path).suffix
        new_image_path = os.path.join(person_folder, image_name + file_ext)
        os.rename(image_path, new_image_path)
    os.rename(person_folder, person_folder.replace(person_name, concat_name))
