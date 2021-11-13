"""
Validate a face recognizer on the "Labeled Faces in the Wild"
(http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted...
"""
import numpy as np
from ..datagen.generator import facematch_datagenerator
from deep_insight_face.datagen.generator import triplet_datagenerator
import os
from typing import Any
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from csv import writer
from . import utility


class TripletEvaluate:
    def __init__(
        self, 
        emd_model, image_paths, pairs,
    ) -> None:
        self.emd_model = emd_model
        self.image_paths = image_paths
        self.pairs = pairs
        
    def __call__(
        self,
        batch_size, nrof_folds, distance_metric,
        subtract_mean=False,
        use_fixed_image_standardization=False,
        use_image_aug_random=False,
        save_output_detail=False
    ):
        # Run forward pass to calculate embeddings

        # embedding_size = int(emd_model.get_shape()[1])
        embedding_size = 128
        generator = triplet_datagenerator(
            self.image_paths, self.pairs,
            do_augment=True,
            batch_size=batch_size,
            target_size=(160, 160),
            n_channels=3,
        )

        nrof_images = len(generator)
        emb_arr = np.zeros((nrof_images, embedding_size))
        lab_arr = np.zeros((nrof_images,))
        paths_arr = []
        idx = 0
        for X_batch, y in next(generator):
            lab = np.arange(idx, idx + len(y))
            idx += len(y)
            emb = self.emd_model.predict_on_batch(X_batch)
            lab_arr[lab] = y
            emb_arr[lab, :] = emb
            paths_arr += generator.pairs[lab]

        assert np.array_equal(lab_arr, np.arange(nrof_images)) is True, \
            'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
        tpr, fpr, accuracy, f1scores, val, val_std, far = utility.evaluate(
            emb_arr, lab_arr, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

        print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
        print("F1 Score: %2.5f+-%2.5f" % (np.mean(f1scores), np.std(f1scores)))
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr, fill_value="extrapolate")(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)

        print('>>>> ============== >>>>')
        if save_output_detail:
            csv_out = result_to_csv(emb_arr, lab_arr, paths_arr, tpr, fpr,
                                    accuracy, f1scores, val, val_std, far)
            csv_out(0.3)


class SiameseEvaluate:
    def __init__(self, emd_model, image_paths, pairs,) -> None:
        self.emd_model = emd_model
        self.image_paths = image_paths
        self.pairs = pairs

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class result_to_csv:
    def __init__(self,
                 emb_arr, lab_arr, paths_arr,
                 tpr, fpr, accuracy, f1scores, val, val_std, far) -> None:
        self.emb_arr, self.lab_arr, self.paths_arr = emb_arr, lab_arr, paths_arr
        self.tpr = tpr
        self.fpr = fpr
        self.accuracy = accuracy
        self.f1scores = f1scores
        self.val = val
        self.val_std = val_std
        self.far = far

    def __call__(self, threshold=0.5, out=None) -> Any:
        outname, ext = os.path.splitext(out)
        result_csv_name = f"{outname}-{threshold}.{ext}"
        results = self.__get_distances__(threshold, result_csv_name)
        return results

    def __get_distances__(self, threshold, result_csv_name) -> Any:
        # TODO: Fix this method
        print('Calculating features for images to check TP OR FN')
        # TP_FN_dict = {}
        with open(result_csv_name, 'w') as csv_file:
            csv_writer = writer(csv_file)
            results = np.array([[0, 0],
                                [0, 0]], dtype=np.int)

            distances = utility.get_emd_distance(self.emb_arr[0], self.emb_arr[1], distance_metric=0)
            results[1][1] += 1
            row = [paths[0].split("/")[-1], paths[1].split("/")[-1], distance, "same", TP_or_FN]
            csv_writer.writerow(row)

            utility.print_confusion_matrix(results, threshold)
        return results
