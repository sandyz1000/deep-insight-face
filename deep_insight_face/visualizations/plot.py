import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

try:
    import seaborn as sns
    sns.set(style="darkgrid")
except ImportError:
    print("Install seaborn package from pip repo")


class pca_visualization:
    """
    ### Visualizing the effect of embeddings -> using PCA
    """

    def __init__(self, x_embeddings, y, classes, no_of_components=2, step=10, epochs=1) -> None:
        self.no_of_components = no_of_components
        self.step = step
        self.epochs = epochs
        self.x_embeddings = x_embeddings
        self.y = y
        self.classes = classes
        self.fig = plt.figure(figsize=(16, 8))

    def __call__(self) -> None:
        pca = PCA(n_components=self.no_of_components)
        decomposed_embeddings = pca.fit_transform(self.x_embeddings[0])
        decomposed_gray = pca.fit_transform(self.x_embeddings[1])

        for label in self.classes:
            decomposed_embeddings_class = decomposed_embeddings[self.y == label]
            decomposed_gray_class = decomposed_gray[self.y == label]

            plt.subplot(1, 2, 1)
            plt.scatter(decomposed_gray_class[::self.step, 1], decomposed_gray_class[::self.step, 0],
                        label=str(label))
            plt.title('before training (embeddings)')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(decomposed_embeddings_class[::self.step, 1], decomposed_embeddings_class[::self.step, 0],
                        label=str(label))
            plt.title('after @%d epochs' % self.epochs)
            plt.legend()

        plt.show()


class tsne_visualization:
    """
    ### Visualization the effect of embeddings -> using TSNE
    """

    def __init__(self) -> None:
        # We choose a color palette with seaborn.
        self.palette = np.array(sns.color_palette("hls", 10))
        self.f = plt.figure(figsize=(8, 8))
        self.ax = plt.subplot(aspect='equal')

    def scatter(self, x, labels, subtitle=None):
        sc = self.ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=self.palette[labels.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        self.ax.axis('off')
        self.ax.axis('tight')

        # We add the labels for each face.
        txts = []
        for i in range(10):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = self.ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        if subtitle:
            plt.suptitle(subtitle)

        plt.savefig(subtitle)

    def __call__(self, x_embeddings, y_v, msg="Training Data After TNN") -> None:
        tsne = TSNE()
        # X_train_trm = trained_model.predict(x_train[:512].reshape(-1, 28, 28, 1))
        train_tsne_embeds = tsne.fit_transform(x_embeddings)
        self.scatter(train_tsne_embeds, y_v, msg)


class hist_plot:
    def __init__(self,
                 history,
                 epochs,
                 names=None,
                 customs=[],
                 save=None,
                 axes=None,
                 init_epoch=0,
                 pre_item={},
                 fig_label=None) -> None:
        self.epochs = epochs
        self.history = history
        self.customs = customs
        self.init_epoch = init_epoch
        self.pre_item = pre_item
        self.fig_label = fig_label
        self.loss_names = names
        self.save = save
        if axes is None:
            self.fig, self.axes = plt.subplots(1, 3, sharex=True, figsize=(24, 8))
        else:
            self.fig = axes[0].figure

    def __arrays_plot__(self, ax, arrays, color=None, label=None, init_epoch=0, pre_value=0):
        tt = []
        for ii in arrays:
            tt += ii
        if pre_value != 0:
            tt = [pre_value] + tt
            xx = list(range(init_epoch, init_epoch + len(tt)))
        else:
            xx = list(range(init_epoch + 1, init_epoch + len(tt) + 1))
        ax.plot(xx, tt, label=label, color=color)
        xticks = list(range(xx[-1]))[:: xx[-1] // 16 + 1]
        # print(xticks, ax.get_xticks())
        if xticks[1] > ax.get_xticks()[1]:
            # print("Update xticks")
            ax.set_xticks(xticks)

    def __peak_scatter__(self, ax, array, peak_method, color="r", init_epoch=0):
        start = init_epoch + 1
        for ii in array:
            pp = len(ii) - peak_method(ii[::-1]) - 1
            ax.scatter(pp + start, ii[pp], color=color, marker="v")
            ax.text(pp + start, ii[pp], "{:.4f}".format(ii[pp]), va="bottom", ha="right", fontsize=8, rotation=-30)
            start += len(ii)

    def plot_histogram(self, loss_lists, accuracy_lists, customs_dict):
        self.__arrays_plot__(self.axes[0], loss_lists, label=self.fig_label,
                             init_epoch=self.init_epoch, pre_value=self.pre_item.get("loss", 0))
        self.__peak_scatter__(self.axes[0], self.loss_lists, np.argmin, init_epoch=self.init_epoch)
        self.axes[0].set_title("loss")
        if self.fig_label:
            self.axes[0].legend(loc="upper right", fontsize=8)

        if len(accuracy_lists) != 0:
            self.__arrays_plot__(self.axes[1], accuracy_lists, label=self.fig_label,
                                 init_epoch=self.init_epoch, pre_value=self.pre_item.get("accuracy", 0))
            self.__peak_scatter__(self.axes[1], accuracy_lists, np.argmax, init_epoch=self.init_epoch)
        self.axes[1].set_title("accuracy")
        if self.fig_label:
            self.axes[1].legend(loc="lower right", fontsize=8)

        # for ss, aa in zip(["lfw", "cfp_fp", "agedb_30"], [lfws, cfp_fps, agedb_30s]):
        for kk, vv in customs_dict.items():
            label = kk + " - " + self.fig_label if self.fig_label else kk
            self.__arrays_plot__(self.axes[2], vv, label=label,
                                 init_epoch=self.init_epoch, pre_value=self.pre_item.get(kk, 0))
            self.__peak_scatter__(self.axes[2], vv, np.argmax, init_epoch=self.init_epoch)
        self.axes[2].set_title(", ".join(customs_dict))
        self.axes[2].legend(loc="lower right", fontsize=8)

        for ax in self.axes:
            ymin, ymax = ax.get_ylim()
            mm = (ymax - ymin) * 0.05
            start = self.init_epoch + 1
            for nn, loss in zip(self.loss_names, self.loss_lists):
                ax.plot([start, start], [ymin + mm, ymax - mm], color="k", linestyle="--")
                # ax.text(xx[ss[0]], np.mean(ax.get_ylim()), nn)
                ax.text(start + len(loss) * 0.05, ymin + mm * 4, nn, va="bottom", rotation=-90)
                start += len(loss)

        self.fig.tight_layout()
        if self.save is not None and len(self.save) != 0:
            self.fig.savefig(self.save)

        last_item = {kk: vv[-1][-1] for kk, vv in customs_dict.items()}
        last_item["loss"] = loss_lists[-1][-1]
        if len(accuracy_lists) != 0:
            last_item["accuracy"] = accuracy_lists[-1][-1]
        return self.axes, self.last_item

    def __call__(self):
        splits = [[int(sum(self.epochs[:id])), int(sum(self.epochs[:id])) + ii]
                  for id, ii in enumerate(self.epochs)]

        def split_func(aa):
            return [aa[ii:jj] for ii, jj in splits if ii < len(aa)]

        if isinstance(self.history, str):
            self.history = [self.history]
        if isinstance(self.history, list):
            import json

            hh = {}
            for pp in self.history:
                with open(pp, "r") as ff:
                    aa = json.load(ff)
                for kk, vv in aa.items():
                    hh.setdefault(kk, []).extend(vv)
            if self.save is not None and len(self.save) == 0:
                self.save = os.path.splitext(pp)[0] + ".svg"
        else:
            hh = self.history.copy()
        loss_lists = split_func(hh.pop("loss"))
        if "accuracy" in hh:
            accuracy_lists = split_func(hh.pop("accuracy"))
        elif "logits_accuracy" in hh:
            accuracy_lists = split_func(hh.pop("logits_accuracy"))
        else:
            accuracy_lists = []
        if len(self.customs) != 0:
            customs_dict = {kk: split_func(hh[kk]) for kk in self.customs if kk in hh}
        else:
            hh.pop("lr")
            customs_dict = {kk: split_func(vv) for kk, vv in hh.items()}
        return self.plot_histogram(loss_lists, accuracy_lists, customs_dict)


def grid_visualization(img: np.ndarray, grid: np.ndarray) -> None:
    """
    Grid Visualization plots

    PLOT for Siamese N/W
    --------------------
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(grid, cmap='gray')
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
