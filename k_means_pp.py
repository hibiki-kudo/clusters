import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class KMeansPlus:

    def __init__(self, classification_num):
        self.classification_num = classification_num
        self.data_sets = None
        self.centroids = []

    def get(function):
        def set(self, data, initial_centroids=False):
            if self.data_sets is None:
                self.data_sets = data
                if initial_centroids is False:
                    data = data.reshape(self.classification_num,
                                        data.shape[0] // self.classification_num,
                                        data.shape[1]
                                        )

            if initial_centroids:
                self.centroids = list(self.set_up_centroid(data))
            else:
                self.centroids = list(self.calc_centroid(data))

        return set

    @get
    def set_data(self, data, initial_centroids=False):
        pass

    def fit(self, epoch=10):
        for _ in range(epoch - 1):
            self.classify()

        return list(self.classify())

    def classify(self):
        result = self.calc_distance_two_point()
        self.centroids = []
        for val in result:
            try:
                points = np.array(val)
                yield [points[:, i] for i in range(len(val[0]))]
            except IndexError:
                pass

        self.set_data(result)

    def calc_centroid(self, data):
        for classification in data:
            classification = np.array(classification)
            centroid = np.array([])
            try:
                for dim in range(len(classification[0])):
                    centroid = np.append(centroid,
                                         np.array([np.sum(classification[:, dim]) /
                                                   np.size(classification[:, dim])]))
            except IndexError:
                continue

            yield centroid

    def set_up_centroid(self, data):
        indexes = np.random.choice(range(len(data)), self.classification_num)
        for index in indexes:
            yield data[index]

    def calc_distance_two_point(self):
        # 分類後の点の位置
        after_classify = [[] for _ in range(self.classification_num)]

        for points in self.data_sets:
            distance = float("inf")
            for j in range(len(self.centroids)):
                if distance >= np.linalg.norm((points - self.centroids[j]), ord=2):
                    distance = np.linalg.norm((points - self.centroids[j]), ord=2)
                    class_index = j
            after_classify[class_index].append(np.array(points))

        return after_classify


if __name__ == "__main__":
    # 分類数
    classification_num = 20
    # 点の数
    set_num = 3000

    # k-meansのインスタンス化
    k_means = KMeansPlus(classification_num)

    # 初期データ設定
    data_set = np.random.normal(10, 10, (set_num, 2))
    # データセットをセット(initial_centroidsをtrueで初期値をkmeans++仕様)
    k_means.set_data(data=data_set, initial_centroids=True)

    # アニメーション用
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax = fig.add_subplot(1, 1, 1)
    image = []
    images = []

    # 生データの描画
    color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                  '#ffff33', '#a65628', '#f781bf', '#0000ff', "#33bb99",
                  "#00ffff", "#556b2e", "#ff00ff", "#2f4f4f", "#b22222",
                  "#800080", "#dc143c", "#b0c4de", ]
    image.append(plt.scatter(data_set[:, 0], data_set[:, 1], alpha=0.8))
    image.append(
        plt.scatter(np.array(k_means.centroids)[:, 0], np.array(k_means.centroids)[:, 1],
                    marker="*", alpha=0.9))
    images.append(image)

    for counter in range(30):
        # k_meansクラスで分類
        after_data = list(k_means.classify())
        image = []
        for i, data in enumerate(after_data):
            image.append(plt.scatter(data[0], data[1], color=color_list[i - 15], alpha=0.8))

        images.append(image)

    data_set = k_means.fit(100)
    image = []
    for i, data in enumerate(data_set):
        image.append(plt.scatter(data[0], data[1], color=color_list[i - 15], alpha=0.8))

    images.append(image)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    ani = animation.ArtistAnimation(fig, images, interval=200)
    ani.save("k_means_plus.gif", writer="imagemagick")
    print(f"宣言時のクラスタ数: {classification_num}\n"
          f"現在のクラスタ数: {len(k_means.centroids)}")
    plt.show()
