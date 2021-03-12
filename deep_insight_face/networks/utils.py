import numpy as np


def distance(emb1, emb2):
    """
    Let's verify on a single triplet example that the squared L2 distance between its
    anchor-positive pair is smaller than the distance between its anchor-negative pair.
    """
    return np.sum(np.square(emb1 - emb2))


def distance_to_proba(distance):
    """Convert distance in the range [0, inf] to probablity [0, 1]
    Arguments:
        distance {float} -- distance scaler
    """
    return 1 / (1 + distance)


def gaussian_kernel_dist_to_prob(distance, tuning_factor=1.0):
    """Convert to probability from distance
        The distance  ||x−x′||  is used in the exponent. The kernel value is in the range [0,1]. 
        There is one tuning parameter σ. Basically if σ is high, K(x,x′) will be close to 1 for any x,x′. 
        If σ is low, a slight distance from x to x′ will lead to K(x,x′) being close to 0
    Arguments:
        distance {float} -- distance scaler
        tuning_factor {float} -- The higher value leads to high probability
    """
    return np.exp(-distance / (2 * tuning_factor**2))


def calc_mean_score(score_dist):
    """Normalize and calculate mean score
    """
    # Normalize here
    score_dist = np.array(score_dist)
    score_dist = score_dist / score_dist.sum()
    # Calculate mean here
    return (score_dist * np.arange(1, 11)).sum()


def set_GPU_limit(limit=2):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * limit)]
            )
            print(f"GPU Limit set to: {1024*limit} MB")
        except RuntimeError as e:
            print(e)

