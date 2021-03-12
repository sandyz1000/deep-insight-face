import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import psutil
import cv2
from .detector.run import YoloDetection, MtcnnDetection

K = tf.compat.v1.keras.backend

USE_MTCNN = True
anchors = np.array([10, 13, 16, 30, 33, 23, 30, 61,
                    62, 45, 59, 119, 116, 90, 156, 198, 373, 326], dtype=np.float)
classes = ["face"]
extensions = ['.jpg', '.jpeg', '.bmp', '.png']
detector_model = None
rotation_model = None
graph = None


def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    angle_difference = (lambda x, y: 180 - abs(abs(x - y) - 180))
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def rotate_image(input_path):
    # TODO: Verify this method
    CPU_COUNT = psutil.cpu_count(logical=False)
    image_paths = []
    if os.path.isfile(input_path):
        image_paths.append(input_path)
    else:
        image_paths += [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in extensions
        ]
    
    def _predict_n_rotate(img_path, angle_rotate=90):
        angle = predict_angle(img_path)
        cropped, bbox = rotate_n_crop(img_path, angle)
        return img_path, angle, cropped, bbox

    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [
            executor.submit(_predict_n_rotate, impath) 
            for impath in image_paths
        ]
        img_paths, predictions, cropped, bbox = concurrent.futures.as_completed(futures)

    return list(zip(img_paths, predictions, cropped, bbox))


def predict_angle(img_path: str):
    img = np.array(Image.open(img_path))
    prediction = rotation_model.predict(img)
    idx = np.argmax(prediction, axis=1)
    return idx


def rotate_n_crop(img_path, predicted_angle, angle_rotate=90):
    image = np.array(Image.open(img_path))
    predicted_angle *= angle_rotate
    rotated_image = rotate(image, -predicted_angle)

    cropped, bbox = detector_model.detect_bbox(rotated_image)
    Image.fromarray(cropped).save(open(img_path, 'wb'), format='PNG')
    return cropped, bbox


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", type=str, help="Source directory")
    parser.add_argument("-d", "--dst_dir", type=str, help="Destination directory")
    parser.add_argument("--rotate", type=bool, default=True, help="Rotate before moving to destination directory")
    parser.add_argument("--crop_n_alignment", type=bool, default=True, help="Crop and align image")
    parser.add_argument("--rotnet_model_path", type=str, required=True, help="Rotation n/w model path")
    parser.add_argument("--detector_model_path", type=str,
                        default='weights/YOLO_Face.h5', help="YOLO detector path")
    parser.add_argument("--detector", choices=["MTCNN", "YOLO"], default="MTCNN", help="Detector type")
    argv = parser.parse_args()
    return argv


def _cli():
    argv = parse_arguments()
    main(argv.detector, argv.detector_model_path, argv.rotnet_model_path)


def main(detector, det_mpath, rotnet_mpath):
    global graph
    global detector_model
    global rotation_model

    assert detector == 'YOLO' and os.path.exists(det_mpath), "Invalid YOLO detector path"
    detector_model = (MtcnnDetection() if detector == 'MTCNN' else
                      YoloDetection(model_path=det_mpath, anchors=anchors, classes=classes))

    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        session = tf.compat.v1.Session()
        K.set_session(session)
        rotation_model = load_model(rotnet_mpath, custom_objects={'angle_error': angle_error})


if __name__ == "__main__":
    _cli()
