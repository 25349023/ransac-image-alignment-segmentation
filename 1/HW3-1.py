import pathlib
import random
from collections import namedtuple
from itertools import chain

import cv2
import numpy as np

SiftFeature = namedtuple('SiftFeature', 'keypoints desc')


def sift_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(0, 7)
    kp, descriptor = sift.detectAndCompute(gray, None)
    return SiftFeature(kp, descriptor)


def sift_matching(left_desc, right_desc, threshold=0.7):
    matches = []
    for q, desc in enumerate(left_desc):
        distance = np.linalg.norm(desc - right_desc, axis=-1)
        sorted_idx = np.argsort(distance)
        t0, t1 = sorted_idx[0:2]
        if distance[t0] / distance[t1] < threshold:
            matches.append([cv2.DMatch(q, t0, distance[t0])])

    return matches


def points_from_matches(matches, left_feat: SiftFeature, right_feat: SiftFeature):
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


def ransac_matching(points, rounds=1000, agree_threshold=10):
    max_agrees = -1
    best_agree_idx = None
    best_H = None
    for _ in range(rounds):
        rand_quad = random_select_quad(points)
        H = solve_homography(*rand_quad)

        left_pts, right_pts = separate_left_right_points(points)
        warpped_left_pts = warp_by(left_pts, H)

        distance = np.linalg.norm(warpped_left_pts - right_pts, axis=1)
        agrees = distance < agree_threshold
        if agrees.sum() > max_agrees:
            max_agrees = agrees.sum()
            best_agree_idx = agrees
            best_H = H

    return best_agree_idx, best_H


def random_select_quad(coords):
    rand_coords = random.choices(coords, k=4)
    rand_coords = list(zip(*rand_coords))
    return rand_coords


def warp_by(pts, H):
    warpped = (H @ pts[..., np.newaxis]).reshape((-1, 3))
    warpped /= warpped[:, 2:]
    return warpped


def separate_left_right_points(points):
    points = np.array(list(zip(*points)))
    book_pts, image_pts = np.concatenate([points, np.ones_like(points)[..., :1]], axis=-1)
    return book_pts, image_pts


def pick_agreed_keypoints(feat, matches, idx_name, agreements):
    matched_kps = [feat.keypoints[getattr(m, idx_name)] for m in matches]
    agreed_kps = [mkp for (agree, mkp) in zip(agreements, matched_kps) if agree]
    return agreed_kps


def deviation_vectors(left_pts, right_pts, H, right_image):
    warpped_left = warp_by(left_pts, H).astype(int)
    right_pts = right_pts.astype(int)
    for left, right in zip(warpped_left, right_pts):
        right_image = cv2.line(right_image, left[:2], right[:2], (0, 230, 255), 1)
    return right_image


if __name__ == '__main__':
    output_dir = pathlib.Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    books = [cv2.imread('1-book1.jpg'), cv2.imread('1-book2.jpg'), cv2.imread('1-book3.jpg')]
    thresholds = [0.8, 0.8, 0.7]
    image = cv2.imread('1-image.jpg')
    image_dev = image.copy()
    image_feat = sift_detect(image)

    for i, (book, thres) in enumerate(zip(books, thresholds)):
        # 1-A
        book_feat = sift_detect(book)
        matches_idx = sift_matching(book_feat.desc, image_feat.desc, thres)

        matches_img = cv2.drawMatchesKnn(
            book, book_feat.keypoints, image, image_feat.keypoints, matches_idx, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0))
        cv2.imwrite(str(output_dir / f'1a_book{i + 1}.jpg'), matches_img)

        # 1-B
        flattened_matches = list(chain.from_iterable(matches_idx))
        pts = points_from_matches(flattened_matches, book_feat, image_feat)

        best_agrees, H = ransac_matching(pts, agree_threshold=3)

        matches_idx = [[cv2.DMatch(i, i, 0)] for i in range(best_agrees.sum())]
        matched_book_kp = pick_agreed_keypoints(book_feat, flattened_matches, 'queryIdx', best_agrees)
        matched_image_kp = pick_agreed_keypoints(image_feat, flattened_matches, 'trainIdx', best_agrees)

        matches_img = cv2.drawMatchesKnn(
            book, matched_book_kp, image, matched_image_kp, matches_idx, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0))
        cv2.imwrite(str(output_dir / f'1b_book{i + 1}.jpg'), matches_img)

        book_pts, image_pts = separate_left_right_points(pts)
        image_dev = deviation_vectors(book_pts[best_agrees], image_pts[best_agrees], H, image_dev)

    cv2.imwrite(str(output_dir / f'1b_deviation_vectors.jpg'), image_dev)

