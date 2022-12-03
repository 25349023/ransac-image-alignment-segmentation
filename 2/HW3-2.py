import itertools
import pathlib
import time

import cv2
import matplotlib
import numpy as np
import tqdm
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, k, init='random', n_guess=10, max_iter=300, epsilon=1e-2):
        self.k_clusters = k
        self.init = init
        self.n_guess = n_guess
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.cluster_centers = None

    @staticmethod
    def norm_square(xs, centers):
        # xs.shape: (n, ch), centers.shape: (k, ch)
        return np.sum(np.square(xs[:, np.newaxis] - centers[np.newaxis]), axis=-1)

    def _init_cluster_centers(self, xs):
        cc = np.array([])
        if self.init == 'k-means':
            idx = np.random.choice(range(xs.shape[0]), self.k_clusters, replace=False)
            cc = xs[idx]
        elif self.init == 'k-means++':
            init_idx = np.random.choice(range(xs.shape[0]))
            cc = np.array([init_idx], dtype=int)
            while len(cc) < self.k_clusters:
                squared_distances = self.norm_square(xs, xs[cc]).min(axis=-1)
                total_dis = squared_distances.sum()
                prob = squared_distances / total_dis
                cc = np.append(cc, np.random.choice(range(xs.shape[0]), p=prob))
            cc = xs[cc]
        return cc

    def predict(self, xs, cluster_centers=None):
        cluster_centers = self.cluster_centers if cluster_centers is None else cluster_centers
        return self.norm_square(xs, cluster_centers).argmin(axis=-1)

    def score(self, cluster_centers, xs):
        distance = self.norm_square(xs, cluster_centers).min(axis=-1).sum()
        return distance

    def _update_center(self, cluster_centers, xs):
        idx = self.predict(xs, cluster_centers)
        new_centers = np.array([xs[idx == i].mean(axis=0) for i in range(self.k_clusters)])
        return new_centers

    def fit(self, xs):
        best, best_cc = -1, None
        for i in range(self.n_guess):
            cluster_centers = self._fit_single_run(xs, i)
            score = self.score(cluster_centers, xs)
            if best == -1 or score < best:
                best, best_cc = score, cluster_centers
        self.cluster_centers = best_cc
        return self

    def _fit_single_run(self, xs, rnd=0):
        cluster_centers = self._init_cluster_centers(xs)
        for _ in tqdm.trange(self.max_iter, desc=f'Round {rnd + 1}: ', leave=False):
            new_cc = self._update_center(cluster_centers, xs)
            new_cc[np.isnan(new_cc)] = cluster_centers[np.isnan(new_cc)]
            change = np.linalg.norm(new_cc - cluster_centers, axis=-1)
            converge = (change < self.epsilon).all()
            if converge:
                break
            cluster_centers = new_cc
        return cluster_centers


class MeanShift:
    def __init__(self, bandwidth, max_iter=100, epsilon=1e-2):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.xs = None
        self.cluster_centers = None

    def predict(self, xs):
        # xs.shape: (n, ch), cluster_centers.shape: (k, ch)
        return np.linalg.norm(xs[:, np.newaxis] - self.cluster_centers[np.newaxis],
                              axis=-1).argmin(axis=-1)

    def fit(self, xs):
        self.xs = xs
        cluster_centers = []
        for x in tqdm.tqdm(xs, leave=False):
            cluster_centers.append(self._fit_one_seed(x))
        self.cluster_centers = cluster_centers
        self.remove_duplicate()
        return self

    def _fit_one_seed(self, x):
        kernel_mean = x
        for _ in range(self.max_iter):
            in_kernel = self._nearby(kernel_mean, self.xs)
            new_kernel = self.xs[in_kernel].mean(axis=0)
            if np.linalg.norm(new_kernel - kernel_mean) < self.epsilon:
                break
            kernel_mean = new_kernel
        return kernel_mean

    def remove_duplicate(self):
        unique_centers = np.unique(np.array(self.cluster_centers, dtype=int), axis=0)
        points_in_kernels = np.array([self._nearby(center, self.xs).sum() for center in unique_centers])
        sorted_centers = unique_centers[points_in_kernels.argsort()[::-1]]

        mark = np.ones(sorted_centers.shape[0], dtype=bool)
        for i, center in enumerate(sorted_centers):
            if mark[i]:
                nearby = self._nearby(center, sorted_centers)
                mark[nearby] = False
                mark[i] = True
        self.cluster_centers = sorted_centers[mark]

    def _nearby(self, center, points):
        distances = np.linalg.norm(center[np.newaxis] - points, axis=-1)
        return distances < self.bandwidth


def plot_pixel_dist(img, alpha=0.3, color='k'):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(img[:, 0], img[:, 1], img[:, 2], alpha=alpha, c=color)
    fig.canvas.draw()

    return get_figure_rgb_data(fig)


def get_figure_rgb_data(fig):
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def combine_with_spatial_info(img):
    h, w, _ = img.shape
    pixels = np.array(list(itertools.product(range(h), range(w))), dtype=float)
    # normalize pixel coordinates to match the dist. of rgb space
    pixels[..., 0] /= h
    pixels[..., 1] /= w
    with_spatial = np.concatenate([img.reshape((-1, 3)), pixels * 255], axis=1)
    return with_spatial


if __name__ == '__main__':
    matplotlib.use('agg')
    output_dir = pathlib.Path('output')

    for name in 'image', 'masterpiece':
        if not (dir := output_dir / name).exists():
            dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(f'2-{name}.jpg')
        flatten_img = image.reshape((-1, 3))

        # 2A, 2B
        for init, k in itertools.product(('k-means', 'k-means++'), (4, 7, 10)):
            print(f'Running {init}, k = {k}')

            start = time.time()
            kmeans = KMeans(k, init=init, n_guess=50, max_iter=100, epsilon=1)
            kmeans.fit(flatten_img)
            print(f'{time.time() - start} secs')

            quantized_image = kmeans.cluster_centers[kmeans.predict(flatten_img, kmeans.cluster_centers)]
            quantized_image = quantized_image.reshape(image.shape).astype(np.uint8)

            cv2.imwrite(str(dir / f'{name}_{init}_{k}.jpg'), quantized_image)

        resized_image = cv2.resize(image, (0, 0), None, 0.3, 0.3)
        flatten_resized_img = resized_image.reshape((-1, 3))

        # 2E
        mean_shifts = []
        segmentations = []
        for bandwidth in 15, 30, 45:
            print(f'Running mean shift, bandwidth = {bandwidth}')

            mean_shift_spatial = MeanShift(bandwidth, max_iter=10, epsilon=1)
            mean_shifts.append(mean_shift_spatial.fit(flatten_resized_img))

            segmentation = mean_shift_spatial.cluster_centers[mean_shift_spatial.predict(flatten_img)]
            segmentation = segmentation.reshape(image.shape).astype(np.uint8)
            segmentations.append(segmentation)

            cv2.imwrite(str(dir / f'2e_{name}_meanshift_{bandwidth}.jpg'), segmentation)

        # 2C
        seg_flat_img = mean_shifts[1].cluster_centers[
                           mean_shifts[1].predict(flatten_resized_img)
                       ].astype(float) / 255

        cv2.imwrite(str(dir / f'2c_{name}_meanshift_clustering_result.jpg'), segmentations[1])
        orig_dist = plot_pixel_dist(flatten_resized_img)
        cv2.imwrite(str(dir / f'2c_{name}_original_pixel_distribution.jpg'), orig_dist)
        seg_dist = plot_pixel_dist(flatten_resized_img, alpha=0.6, color=seg_flat_img)
        cv2.imwrite(str(dir / f'2c_{name}_segmentation_pixel_distribution.jpg'), seg_dist)

        # 2D
        print(f'Running mean shift with spatial info')
        resized_with_spatial = combine_with_spatial_info(resized_image)
        img_with_spatial = combine_with_spatial_info(image)

        mean_shift_spatial = MeanShift(40, max_iter=10, epsilon=3)
        mean_shift_spatial.fit(resized_with_spatial)

        segmentation = mean_shift_spatial.cluster_centers[mean_shift_spatial.predict(img_with_spatial)]
        segmentation = segmentation[..., :3].reshape(image.shape).astype(np.uint8)
        cv2.imwrite(str(dir / f'2d_{name}_meanshift_with_spatial_info.jpg'), segmentation)
