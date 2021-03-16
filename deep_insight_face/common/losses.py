import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops


class TripletLossWapper(tf.keras.losses.Loss):
    """
    TripletLoss helper class definitions [Triplet Loss and Online Triplet Mining in TensorFlow]
    (https://omoindrot.github.io/triplet-loss) 

    """    
    def __init__(self, alpha=0.35, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(TripletLossWapper, self).__init__(**kwargs, reduction=reduction)
        super(TripletLossWapper, self).__init__(**kwargs)
        self.alpha = alpha

    def __calculate_triplet_loss__(self, y_true, y_pred, alpha):
        return None

    def call(self, labels, embeddings):
        return self.__calculate_triplet_loss__(labels, embeddings, self.alpha)

    def get_config(self):
        config = super(TripletLossWapper, self).get_config()
        config.update({"alpha": self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BatchHardTripletLoss(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        # labels = tf.squeeze(labels)
        # labels.set_shape([None])
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        hardest_pos_dist = tf.reduce_min(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * -1, dists)
        hardest_neg_dist = tf.reduce_max(neg_dists, -1)
        basic_loss = hardest_neg_dist - hardest_pos_dist + alpha
        # ==> pos - neg > alpha
        # ==> neg + alpha - pos < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


class BatchHardTripletLossEuclidean(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        # dense_mse_func = lambda xx: tf.reduce_sum(tf.square((embeddings - xx)), axis=-1)
        # dense_mse_func = tf.function(dense_mse_func, input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
        # dists = tf.vectorized_map(dense_mse_func, embeddings)

        # Euclidean_dists = aa ** 2 + bb ** 2 - 2 * aa * bb, where aa = embeddings, bb = embeddings
        embeddings_sqaure_sum = tf.reduce_sum(tf.square(embeddings), axis=-1)
        ab = tf.matmul(embeddings, tf.transpose(embeddings))
        dists = tf.reshape(embeddings_sqaure_sum, (-1, 1)) + embeddings_sqaure_sum - 2 * ab
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.zeros_like(dists))
        hardest_pos_dist = tf.reduce_max(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * tf.reduce_max(dists), dists)
        hardest_neg_dist = tf.reduce_min(neg_dists, -1)
        tf.print(
            " - triplet_dists_mean:",
            tf.reduce_mean(dists),
            "pos:",
            tf.reduce_mean(hardest_pos_dist),
            "neg:",
            tf.reduce_mean(hardest_neg_dist),
            end="",
        )
        basic_loss = hardest_pos_dist + alpha - hardest_neg_dist
        # ==> neg - pos > alpha
        # ==> pos + alpha - neg < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


class BatchHardTripletLossEuclideanAutoAlpha(TripletLossWapper):
    def __init__(self, alpha=0.1, init_auto_alpha=1, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(TripletLossWapper, self).__init__(**kwargs, reduction=reduction)
        super(BatchHardTripletLossEuclideanAutoAlpha, self).__init__(alpha=alpha, **kwargs)
        self.auto_alpha = tf.Variable(init_auto_alpha, dtype="float", trainable=False)

    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        # dense_mse_func = lambda xx: tf.reduce_sum(tf.square((embeddings - xx)), axis=-1)
        # dense_mse_func = tf.function(dense_mse_func, input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),))
        # dists = tf.vectorized_map(dense_mse_func, embeddings)

        # Euclidean_dists = aa ** 2 + bb ** 2 - 2 * aa * bb, where aa = embeddings, bb = embeddings
        embeddings_sqaure_sum = tf.reduce_sum(tf.square(embeddings), axis=-1)
        ab = tf.matmul(embeddings, tf.transpose(embeddings))
        dists = tf.reshape(embeddings_sqaure_sum, (-1, 1)) + embeddings_sqaure_sum - 2 * ab
        # pos_dists = tf.ragged.boolean_mask(dists, pos_mask)
        pos_dists = tf.where(pos_mask, dists, tf.zeros_like(dists))
        hardest_pos_dist = tf.reduce_max(pos_dists, -1)
        # neg_dists = tf.ragged.boolean_mask(dists, tf.logical_not(pos_mask))
        neg_dists = tf.where(pos_mask, tf.ones_like(dists) * tf.reduce_max(dists), dists)
        hardest_neg_dist = tf.reduce_min(neg_dists, -1)
        basic_loss = hardest_pos_dist + self.auto_alpha - hardest_neg_dist
        self.auto_alpha.assign(tf.reduce_mean(dists) * alpha)
        tf.print(
            " - triplet_dists_mean:",
            tf.reduce_mean(dists),
            "pos:",
            tf.reduce_mean(hardest_pos_dist),
            "neg:",
            tf.reduce_mean(hardest_neg_dist),
            "auto_alpha:",
            self.auto_alpha,
            end="",
        )
        # ==> neg - pos > alpha
        # ==> pos + alpha - neg < 0
        # return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return tf.maximum(basic_loss, 0.0)


class BatchAllTripletLoss(TripletLossWapper):
    def __calculate_triplet_loss__(self, labels, embeddings, alpha):
        labels = tf.argmax(labels, axis=1)
        # labels = tf.squeeze(labels)
        # labels.set_shape([None])
        pos_mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        norm_emb = tf.nn.l2_normalize(embeddings, 1)
        dists = tf.matmul(norm_emb, tf.transpose(norm_emb))

        pos_dists = tf.where(pos_mask, dists, tf.ones_like(dists))
        pos_dists_loss = tf.reduce_sum(1.0 - pos_dists, -1) / tf.reduce_sum(tf.cast(pos_mask, dtype=tf.float32), -1)
        hardest_pos_dist = tf.expand_dims(tf.reduce_min(pos_dists, -1), 1)

        neg_valid_mask = tf.logical_and(tf.logical_not(pos_mask), (hardest_pos_dist - dists) < alpha)
        neg_dists_valid = tf.where(neg_valid_mask, dists, tf.zeros_like(dists))
        neg_dists_loss = tf.reduce_sum(neg_dists_valid, -1) / \
            (tf.reduce_sum(tf.cast(neg_valid_mask, dtype=tf.float32), -1) + 1)
        return pos_dists_loss + neg_dists_loss


def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.

    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    # dot_product = math_ops.matmul(embeddings, array_ops.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    # square_norm = array_ops.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)

    # pairwise_distances_squared = array_ops.expand_dims(square_norm, 0) \
    #     - 2.0 * dot_product \
    #     + array_ops.expand_dims(square_norm, 1)

    pairwise_distances_squared = math_ops.add(
        math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
        math_ops.reduce_sum(math_ops.square(array_ops.transpose(feature)), axis=[0], keepdims=True))
    - 2.0 * math_ops.matmul(feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        pairwise_distances = math_ops.sqrt(
            pairwise_distances_squared + tf.cast(math_ops.to_float(error_mask) * 1e-16, dtype=tf.float32))

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
        pairwise_distances, tf.cast(math_ops.to_float(math_ops.logical_not(error_mask)), dtype=tf.float32))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - \
        tf.cast(array_ops.diag(array_ops.ones([num_data])), tf.float32)
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the maximum.

    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
    masked_maximums = math_ops.reduce_max(
        math_ops.multiply(data - axis_minimums, tf.cast(mask, dtype=tf.float32)), dim,
        keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.

    Args:
      data: 2-D float `Tensor` of size [n, m].
      mask: 2-D Boolean `Tensor` of size [n, m].
      dim: The dimension over which to compute the minimum.

    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
    masked_minimums = math_ops.reduce_min(
        math_ops.multiply(data - axis_maximums, tf.cast(mask, dtype=tf.float32)), dim,
        keepdims=True) + axis_maximums
    return masked_minimums


def triplet_loss_adapted_from_tf(labels, embeddings):
    margin = 1
    labels = tf.argmax(labels, axis=1)
    # Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:

    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    # lshape=array_ops.shape(labels)
    # assert lshape.shape == 1
    # labels = array_ops.reshape(labels, [lshape[0], 1])
    labels = array_ops.reshape(labels, [-1, 1])
    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    # global batch_size
    batch_size = array_ops.size(labels)  # was 'array_ops.size(labels)'

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
        array_ops.tile(adjacency_not, [batch_size, 1]),
        math_ops.greater(pdist_matrix_tile, array_ops.reshape(array_ops.transpose(pdist_matrix), [-1, 1]))
    )
    mask_final = array_ops.reshape(
        math_ops.greater(math_ops.reduce_sum(math_ops.cast(mask, dtype=tf.float32), 1, keepdims=True), 0.0),
        [batch_size, batch_size]
    )

    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=tf.float32)
    mask = math_ops.cast(mask, dtype=tf.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(adjacency, dtype=tf.float32) - \
        math_ops.cast(array_ops.diag(array_ops.ones([batch_size])), dtype=tf.float32)

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    semi_hard_triplet_loss_distance = math_ops.truediv(
        math_ops.reduce_sum(math_ops.maximum(math_ops.multiply(loss_mat, mask_positives), 0.0)),
        math_ops.cast(num_positives, dtype=tf.float32),
        name='triplet_semihard_loss')

    # Code from Tensorflow function semi-hard triplet loss ENDS here.
    return semi_hard_triplet_loss_distance
