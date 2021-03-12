import os
import cv2
import random

import logging
import inspect
from itertools import islice
from functools import wraps
import dill as pickle
from threading import Lock

logger = logging.getLogger(__name__)


def thread_safe_singleton(cls):
    # Decorator routine to create single instance of an class
    instances = {}
    session_lock = Lock()
    
    @wraps
    def wrapper(*args, **kwargs):
        with session_lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


def thread_safe_memoize(func):
    # Decorator routine to create single instance of an function value
    cache = {}
    session_lock = Lock()

    @wraps
    def memoizer(*args, **kwargs):
        with session_lock:
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


def get_weight_path(model_dir_path, model_name):
    return model_dir_path + os.path.sep + model_name + '-weights.h5'


def custom_memoize(func):
    memo = {}

    @wraps
    def wrap_function(*args, **kwargs):
        serialize_args = pickle.dumps(args) if len(args) > 0 else pickle.dumps(func.__name__)
        if serialize_args not in memo:
            memo[serialize_args] = func(args, kwargs)
        return memo[serialize_args]
    return wrap_function


def cus_mkdirs(path, check=True):
    if check and not os.path.exists(path):
        os.makedirs(path)
        return True


def get_all_class(module, only_child=False, parent=None):
    def is_class_member(member):
        is_class = inspect.isclass(member) and member.__module__ == module.__name__
        return True if is_class and only_child and issubclass(member, parent) else False
    clsmembers = inspect.getmembers(module, is_class_member)
    return clsmembers


def split_dict_equally(input_dict, partitions=2):
    it = iter(input_dict)
    for i in range(0, len(input_dict), partitions):
        yield {k: input_dict[k] for k in islice(it, partitions)}


def get_random_image(listfiles, filename, path, suffix=None):
    filesep = "_"
    random_file = random.choice(listfiles)
    # TODO: Fix kyc_name helper method
    kyc_name = os.path.splitext(random_file.split(filesep)[-1])[0]
    while filename.split(filesep)[0] == random_file.split(filesep)[0] and kyc_name != suffix:
        random_file = random.choice(listfiles)
        kyc_name = os.path.splitext(random_file.split(filesep)[-1])[0]

    return os.path.join(path, random_file)


def save_failed_images(path, path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_file_Path = os.path.join(path_to_save, path.split(os.sep)[-1])
    # print("PATH ",save_file_Path)
    try:
        cv2.imwrite(save_file_Path, image)
    except Exception as e:
        print("Excep", e)


def get_dyanamic_image_path(top, filename, suffix):
    is_file_exists = (lambda path_to_file: True if os.path.isfile(path_to_file) else False)
    dynamic_path = os.path.join(top, filename.split('_')[0] + '_' + filename.split('_')[1] + '_' +
                                filename.split('_')[2] + '_' + filename.split('_')[3] + '_' + suffix + '.' +
                                filename.split('.')[-1])
    return dynamic_path if is_file_exists(dynamic_path) else None


