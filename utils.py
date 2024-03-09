import numpy as np
import cv2
import math
from itertools import combinations


def getGridLines(img: np.ndarray, minLineLength: int = 35, maxLineGap: int = 10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(
        gray, 11, 17, 17, borderType=cv2.BORDER_CONSTANT)

    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 3)

    adaptive = cv2.morphologyEx(
        adaptive, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(adaptive, 30, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=minLineLength, maxLineGap=maxLineGap)

    lines = lines.squeeze()

    return lines


def filterLines(lines, threshold=0.1):
    filtered = []
    skip_lines = {-1}

    for i, line1 in enumerate(lines):
        if i in skip_lines:
            continue

        m1 = slope(line1)

        for j, line2 in enumerate(lines):
            m2 = slope(line2)

            if i == j:
                continue

            if math.isclose(m1, m2, abs_tol=threshold):
                filtered.append(line1)

    # filtered2 = []

    # for line in filtered:
    #     m1 = slope(line)
    #     for line2 in filtered2:
    #         m2 = slope(line2)
    #         if math.isclose(m1, m2, abs_tol=0.1) and isLineClose(line, line2, m=m1):
    #             break
    #     else:
    #         filtered2.append(line)

    # return filtered2
    return filtered


def isLineClose(line1, line2, threshold=10, m=None):
    return getDistance(line1, line2, m=m) <= threshold


def slope(line):
    x1, y1, x2, y2 = line

    if x2 - x1 == 0:
        return np.inf

    return (y2 - y1) / (x2 - x1)


def getDistance(line1, line2, m=None):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if m is None:
        m = slope(line1)

    if m == np.inf:
        return abs(x3 - x1)

    return abs((y3 - m * x3) - (y1 - m * x1)) / np.sqrt(m**2 + 1)


def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    if x1 == x2 and x3 == x4:
        return None

    if x1 == x2:
        m2 = slope(line2)
        x = x1
        y = m2 * x + y3 - m2 * x3
        return (x, y)

    if x3 == x4:
        m1 = slope(line1)
        x = x3
        y = m1 * x + y1 - m1 * x1
        return (x, y)

    m1 = slope(line1)
    m2 = slope(line2)

    if m1 == m2:
        return None

    x = (y3 - y1 - m2 * x3 + m1 * x1) / (m1 - m2)
    y = m1 * x + y1 - m1 * x1

    return (x, y)


def slopesOfOutline(contours, threshold=5):
    lines = combinations(contours, 2)

    slopes = [slope([pt1[0], pt1[1], pt2[0], pt2[1]]) for (pt1, pt2) in lines]

    approx = []

    visited = set()

    # filter 1
    for i, m in enumerate(slopes):
        if i in visited:
            continue

        angle1 = math.atan(m) * 180 / np.pi
        similar = []
        for j, m2 in enumerate(slopes):
            if j in visited:
                continue

            if j == i:
                continue

            angle2 = math.atan(m2) * 180 / np.pi

            if math.isclose(angle1, angle2, abs_tol=threshold):
                similar.append(m2)
                visited.add(j)

        if len(similar) > 0:
            total = sum(similar) + m
            if total == np.inf:
                approx.append(np.inf)
            else:
                approx.append(total / (len(similar) + 1))
        else:
            approx.append(m)

    # filter 2
    slope_pair = combinations(approx, 2)

    for m in approx:
        for (m1, m2) in slope_pair:

            if m == m1 or m == m2:
                continue

            if math.isclose(m, (m1 + m2)/2, abs_tol=0.2):
                approx.remove(m)
                break

    # filter 3
    for i, m in enumerate(approx):
        found = False

        for j, m2 in enumerate(approx):
            if i == j:
                continue

            if math.isclose(m*m2, -1, abs_tol=0.2):
                found = True
                break

        if not found:
            approx.remove(m)

    return approx


def dp_simplify(points, epsilon):
    def point_dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def find_furthest_point(p0, p1, p2, points):
        max_dist = -1
        max_point = None
        for p in points:
            if p == p0 or p == p1 or p == p2:
                continue
            dist = point_dist(p, p1)
            if dist > max_dist:
                max_dist = dist
                max_point = p
        return max_point

    def simplify(points, epsilon, start_index, end_index):
        if end_index - start_index < 2:
            return []
        p0 = points[start_index]
        p1 = points[end_index]
        max_dist = -1
        max_point = None
        for i in range(start_index + 1, end_index):
            p = points[i]
            dist = point_dist(p, p0)
            if dist > max_dist:
                max_dist = dist
                max_point = p
        if max_dist < epsilon:
            return [p0, p1]
        else:
            furthest_point = find_furthest_point(p0, p1, max_point, points)
            return (
                simplify(points, epsilon, start_index,
                         points.index(furthest_point))
                + simplify(points, epsilon,
                           points.index(furthest_point), end_index)
            )

    return simplify(points, epsilon, 0, len(points) - 1)
