import numpy as np
from typing import Any
import time
import matplotlib.pyplot as plt
from .visualizations import plot


# TODO: FIX THIS MODULE
def make_oneshot_task(features: np.ndarray, categories, num_class: int, language: Any = None):
    """
    Create pairs of test image, support set for testing N way one-shot learning.

    Arguments:
        features {[type]} -- [description]
        categories {[type]} -- [description]
        num_class {[type]} -- how many classes for testing one-shot tasks

    Keyword Arguments:
        language {[Any]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    n_classes, n_examples, w, h = features.shape

    indices = np.random.randint(0, n_examples, size=(num_class,))
    if language is not None:  # if language is specified, select characters for that language
        low, high = categories[language]
        if num_class > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, num_class))
        categories = np.random.choice(range(low, high), size=(num_class,), replace=False)

    else:  # if no language specified just pick a bunch of random letters
        categories = np.random.randint.choice(range(n_classes), size=(num_class,), replace=False)
    true_category = categories[0]
    ex1, ex2 = np.random.randint.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([features[true_category, ex1, :, :]] * num_class).reshape(num_class, w, h, 1)
    support_set = features[categories, indices, :, :]
    support_set[0, :, :] = features[true_category, ex2]
    support_set = support_set.reshape(num_class, w, h, 1)
    targets = np.zeros((num_class, ))
    targets[0] = 1
    targets, test_image, support_set = np.random.shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def get_batch(batch_size, s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = np.random.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = np.random.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + np.random.randint(1, n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def init_training():
    N_way = 20  # how many classes for testing one-shot tasks
    n_val = 250  # how many one-shot tasks to validate on
    best = -1

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    for i in range(1, n_iter + 1):
        (inputs, targets) = get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time() - t_start) / 60.0))
            print("Train Loss: {0}".format(loss))
            val_acc = test_oneshot(model, N_way, n_val, verbose=True)
            model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(i)))
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc


def one_shot_clf():
    # refer from Siamese_on_Omniglot_Dataset.py
    """ One-shot classifier Evaluation
    """
    # ----------------------------
    # ONE-SHOT Classifier
    # ----------------------------

    # Example of concat image visualization

    pairs, targets = make_oneshot_task(16, "train", "Sanskrit")
    pairs[0][0].reshape(105, 105)
    img = concat_images(pairs[1])
    # plot_oneshot_task(img)
    plot.grid_visualization(img)

    fig, ax = plt.subplots(1)
    ax.plot(ways, val_accs, "m", label="Siamese(val set)")
    ax.plot(ways, train_accs, "y", label="Siamese(train set)")
    plt.plot(ways, nn_accs, label="Nearest neighbour")

    ax.plot(ways, 100.0 / ways, "g", label="Random guessing")
    plt.xlabel("Number of possible classes in one-shot tasks")
    plt.ylabel("% Accuracy")
    plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    inputs, targets = make_oneshot_task(20, "val", 'Oriya')
    plt.show()

    plot_oneshot_task(inputs)
