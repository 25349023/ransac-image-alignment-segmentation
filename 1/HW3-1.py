import pathlib
import random
from collections import namedtuple
from itertools import chain

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

SiftFeature = namedtuple('SiftFeature', 'keypoints desc')


def sift_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(0, 7)
    kp, descriptor = sift.detectAndCompute(gray, None)
    print(len(kp))
    return SiftFeature(kp, descriptor)


def sift_matching(left_desc, right_desc, threshold=200):
    matches = []
    for q, desc in enumerate(left_desc):
        distance = np.linalg.norm(desc - right_desc, axis=-1)
        min_idx = np.argmin(distance)
        if distance[min_idx] < threshold:
            matches.append([cv2.DMatch(q, min_idx, distance[min_idx])])

    return matches


def coords_from_matches(matches, left_feat: SiftFeature, right_feat: SiftFeature):
    return [[left_feat.keypoints[m.queryIdx].pt, right_feat.keypoints[m.trainIdx].pt]
            for m in matches]


def relation_matrix(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.array(
        [[x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2],
         [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]]
    )


def solve_homography(pts1, pts2):
    a_matrix = np.vstack([relation_matrix(p, q) for p, q in zip(pts1, pts2)])
    _, _, vt = np.linalg.svd(a_matrix)
    x = vt[-1].reshape((3, 3))
    return x


if __name__ == '__main__':
    matplotlib.use('agg')

    output_dir = pathlib.Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    books = [cv2.imread('1-book1.jpg'), cv2.imread('1-book2.jpg'), cv2.imread('1-book3.jpg')]
    thresholds = [283, 200, 301]
    image = cv2.imread('1-image.jpg')
    image_feat = sift_detect(image)

    for i, (book, thres) in enumerate(zip(books, thresholds)):
        book_feat = sift_detect(book)

        matches_idx = sift_matching(book_feat.desc, image_feat.desc, thres)
        matches_img = cv2.drawMatchesKnn(
            book, book_feat.keypoints, image, image_feat.keypoints, matches_idx, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0))
        cv2.imwrite(str(output_dir / f'1a_book{i + 1}.jpg'), matches_img)

