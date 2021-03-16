# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import typing
import warnings
from .networks.utils import gaussian_kernel_dist_to_prob, distance_to_proba
from .training import get_embedding
from .networks.triplet import bottleneck_network as triplet_bottleneck
from .networks.siamese import bottleneck_network as siamese_bottleneck
from easydict import EasyDict
from .exceptions.face_exception import FaceRecognitionException
from .config import cfg
try:
    from face_landmark_detector.landmark import detect_marks
except ImportError:
    print(
        """
        Install face-landmark-detector module to use the facial keypoints detector
        pip install git+
        """
    )
    sys.exit(1)

INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

warnings.simplefilter("ignore")


def _face_recognition_model(self, modelname: FACEM_MODEL, args: EasyDict = None,
                            model_path: str = None) -> typing.Any:
    assert model_path is not None, "Invalid Face recognition model path"
    model = None
    if modelname == FACEM_MODEL.SIAMESE:
        args = EasyDict(net=cfg.FACERECO_CFG.BOTTLENECK,
                        emb_size=cfg.FACERECO_CFG.EMD_SIZE, input_shape=cfg.FACERECO_CFG.IMG_SIZE,
                        use_keras_bn=True, summary=True)
        model = siamese_bottleneck(**args)()
    elif modelname == FACEM_MODEL.TRIPLET:
        args = EasyDict(net=cfg.FACERECO_CFG.BOTTLENECK,
                        emb_size=cfg.FACERECO_CFG.EMD_SIZE, input_shape=cfg.FACERECO_CFG.IMG_SIZE,
                        use_keras_bn=True, summary=True)
        model = triplet_bottleneck(**args)()
    else:
        raise ValueError("Model Name invalid")
    model.load_weights(model_path)
    return model


face_recognition_api = _face_recognition_model()


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean
    distance for each comparison face. The distance tells you how similar the faces are.

    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    dist = np.linalg.norm(face_encodings - face_to_compare, axis=0)
    return dist


def _raw_face_landmarks(face_image, face_locations):

    if face_locations is None:
        thumb, face_locations = face_recognition_api.detect_bbox(face_image)

    bb = getLargestFaceBoundingBox(face_locations)
    return findLandmarks(face_image, bb, face_recognition_api.pose_pred_model)


def getLargestFaceBoundingBox(face_locations, skipMulti=False):
    """ Find the largest face bounding box in an image. """
    assert face_locations is not None

    if (not skipMulti and len(face_locations) > 0) or len(face_locations) == 1:
        return [max(face_locations, key=lambda rect: abs(rect[2] - rect[0]) * abs(rect[3] - rect[1]))]


def findLandmarks(rgbImg, rects, pose_predictor_model):
    """ Find the landmarks of a face. """
    assert rgbImg is not None
    assert rects is not None
    landmarks = [detect_marks(rgbImg, pose_predictor_model, rect) for rect in rects]
    return landmarks


def create_thumbnail(rgbImg, points, imgDim=96):
    from .utils.filehelper import save_img
    assert imgDim is not None
    assert rgbImg is not None

    landmarks = list(map(lambda pt: (pt[0], pt[1]), points))
    landmarkIndices = INNER_EYES_AND_BOTTOM_LIP
    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)

    H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices], imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
    thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))
    save_img(thumbnail)
    return thumbnail


def face_landmarks(face_image, face_locations=None, model='large'):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in
    the image
    ### For a definition of each point index,
    ### see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)

    if model == 'large':
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] +
            [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] +
            [points[67]] + [points[66]] + [points[65]] + [points[64]]
        } for points in landmarks]
    elif model == 'small':
        return [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks]
    else:
        raise ValueError("Invalid landmarks model type. Supported models are ['small', 'large'].")


def detect_and_alignment(img: typing.Union[np.ndarray, str], **kwargs):
    """
    ### DETECT NUMPY/PIL IMAGE AND ALIGN ###
    """

    thumbs, rects = face_recognition_api.detect_bbox(img)
    landmarks = None
    # landmarks = findLandmarks(img, rects, face_recognition_api.pose_pred_model)
    # thumbs = [create_thumbnail(thumb, landmark) for thumb, landmark in zip(thumbs, landmarks)]

    if kwargs.get('do_show_plot', False):
        show_plot(img, thumbs, rects, prefix="face")
    return thumbs, rects, landmarks


def face_encodings(face_image: np.ndarray,
                   image_size: typing.Tuple,
                   do_show_plot: bool = False,
                   detect_and_crop: bool = True):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.
    It used keras to predict the encoding
    """
    thumb = [face_image]

    try:
        if detect_and_crop:
            thumb, _, _ = detect_and_alignment(face_image, do_show_plot=do_show_plot)
    except (ValueError, FaceRecognitionException) as e:
        raise e

    try:
        encoding = get_embedding(
            face_recognition_api.module.value,
            face_recognition_api.frmodel, thumb[0], image_size=image_size)
    except FaceRecognitionException as e:
        raise e

    return thumb, encoding


def show_plot(jc_orig, jc_aligned, bb, prefix="dummy"):
    from tempfile import NamedTemporaryFile
    if jc_orig is not None:
        plt.subplot(132)
        with NamedTemporaryFile(prefix=prefix, suffix='jc_orig_box') as f:
            fullpath = f.name + ".png"
            bb = getLargestFaceBoundingBox(bb, skipMulti=False)
            cv2.rectangle(jc_orig, (bb[0], bb[1]), (bb[2], bb[3]), color=(0, 0, 255), thickness=3)
            plt.imsave(fullpath, jc_orig)

    if jc_aligned is not None:
        plt.subplot(133)
        with NamedTemporaryFile(prefix=prefix, suffix='jc_aligned') as f:
            # jc_aligned = np.transpose(jc_aligned, (1, 2, 0))
            fullpath = f.name + ".png"
            plt.imsave(fullpath, jc_aligned)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more
    strict. 0.6 is typical best performance.
    """
    distance = face_distance(known_face_encodings[0], face_encoding_to_check[0])
    if distance <= tolerance:
        probability = gaussian_kernel_dist_to_prob(distance)
    else:
        probability = distance_to_proba(distance)
    return distance, probability
