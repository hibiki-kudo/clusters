import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class KMeansPlus:

    def __init__(self, classification_num):
        self.classification_num = classification_num
        self.data_sets = None
        self.centroids = []

    def get(function):
        def set(self, x, y, initial_centroids=False):
            if self.data_sets is not None:
                self.data_sets += [[x, y]]
            else:
                self.data_sets = [[x, y]]

            if initial_centroids:
                self.centroids += list(self.set_up_centroid(x, y))
            else:
                x, y = self.calc_centroid(x, y)
                self.centroids += [[x, y]]

        return set

    @get
    def set_data(self, x, y, initial_centroids=False):
        pass

    def classify(self):
        result = self.calc_distance_two_point()
        self.data_sets = None
        self.centroids = []
        for val in result:
            try:
                points = np.array(val)
                x = points[:, 0]
                y = points[:, 1]
                self.set_data(x, y, initial_centroids=False)
                yield x, y
            except IndexError:
                pass

    @staticmethod
    def calc_centroid(x, y):
        centroid_x = np.sum(x) / np.size(x)
        centroid_y = np.sum(y) / np.size(y)

        return centroid_x, centroid_y

    @staticmethod
    def set_up_centroid(x, y):
        indexes = np.random.choice(range(len(x)), 1)
        for index in indexes:
            yield [x[index], y[index]]

    def calc_distance_two_point(self):
        # 分類後の点の位置
        after_classify = [[] for _ in range(self.classification_num)]

        for points in self.data_sets:
            for i in range(len(points[0])):
                distance = float("inf")
                for j in range(len(self.centroids)):
                    if distance >= np.linalg.norm((points[0][i] - self.centroids[j][0],
                                                   points[1][i] - self.centroids[j][1]), ord=2):
                        distance = np.linalg.norm((points[0][i] - self.centroids[j][0],
                                                   points[1][i] - self.centroids[j][1]), ord=2)
                        class_index = j
                after_classify[class_index].append([points[0][i], points[1][i]])

        return after_classify


if __name__ == "__main__":
    # 分類数
    classification_num = 10
    # 点の数
    set_num = 3000

    # k-meansのインスタンス化
    k_means = KMeansPlus(classification_num)

    # アニメーション用
    fig = plt.figure()
    image = []
    images = []

    # 生データの描画
    color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                  '#ffff33', '#a65628', '#f781bf', '#0000ff', "#33bb99"]
    for i in range(classification_num):
        x = np.random.randn(set_num // classification_num)
        y = np.random.randn(set_num // classification_num)
        k_means.set_data(x=x, y=y, initial_centroids=True)
        image.append(plt.scatter(x=x, y=y, color=color_list[i], alpha=0.8))

    images.append(image)

    for counter in range(20):
        # k_meansクラスで分類
        after_data = list(k_means.classify())
        image = []
        for i, data in enumerate(after_data):
            image.append(plt.scatter(x=data[0], y=data[1], color=color_list[i], alpha=0.8))
            # plt.scatter(k_means.centroids[i][0], k_means.centroids[i][1], s=100, marker="*", label=f"重心{i+1}")
        images.append(image)
        # plt.grid(True)
        # plt.legend(loc='upper left')

    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ani = animation.ArtistAnimation(fig, images, interval=200)
    ani.save("k_means_plus.gif", writer="imagemagick")
    print(len(k_means.centroids))
    plt.show()
