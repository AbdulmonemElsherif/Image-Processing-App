import cv2
import numpy as np
import random
from skimage.feature import canny
import matplotlib.pyplot as plt

def randomized_hough_ellipse_transform(img, max_iter=1000, major_axis_bound=(50, 1000), minor_axis_bound=(50, 1000),
                                       max_flattening=0.3, canny_sigma=4, canny_t1=20, canny_t2=50, 
                                       similar_center_dist=10, similar_major_axis_dist=10, similar_minor_axis_dist=10, 
                                       similar_angle_dist=np.pi/18):
    
    def canny_edge_detector(image, sigma, low_threshold, high_threshold):
        edged_image = canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        edge = np.zeros(image.shape, dtype=np.uint8)
        edge[edged_image] = 255
        return edge

    def randomly_pick_point(edge_pixels):
        return random.sample(edge_pixels, 3)

    def find_center(edge, pt):
        def point_out_of_image(point):
            if point[0] < 0 or point[0] >= edge.shape[1] or point[1] < 0 or point[1] >= edge.shape[0]:
                raise ValueError("center is out of image!")

        m_arr, c_arr = [], []
        for i in range(len(pt)):
            xstart, xend = max(pt[i][0] - 3, 0), min(pt[i][0] + 4, edge.shape[1])
            ystart, yend = max(pt[i][1] - 3, 0), min(pt[i][1] + 4, edge.shape[0])
            crop = edge[ystart: yend, xstart: xend].T
            proximal_point = np.array(np.where(crop == 255)).T
            proximal_point[:, 0] += xstart
            proximal_point[:, 1] += ystart
            A = np.vstack([proximal_point[:, 0], np.ones(len(proximal_point[:, 0]))]).T
            if len(proximal_point) == 0:
                raise ValueError("No proximal points found")
            m, c = np.linalg.lstsq(A, proximal_point[:, 1], rcond=None)[0]
            m_arr.append(m)
            c_arr.append(c)
        slope_arr, intercept_arr = [], []
        for i, j in zip([0, 1], [1, 2]):
            coef_matrix = np.array([[m_arr[i], -1], [m_arr[j], -1]])
            dependent_variable = np.array([-c_arr[i], -c_arr[j]])
            t12 = np.linalg.solve(coef_matrix, dependent_variable)
            m1 = ((pt[i][0] + pt[j][0]) / 2, (pt[i][1] + pt[j][1]) / 2)
            slope = (m1[1] - t12[1]) / (m1[0] - t12[0])
            intercept = (m1[0] * t12[1] - t12[0] * m1[1]) / (m1[0] - t12[0])
            slope_arr.append(slope)
            intercept_arr.append(intercept)
        coef_matrix = np.array([[slope_arr[0], -1], [slope_arr[1], -1]])
        dependent_variable = np.array([-intercept_arr[0], -intercept_arr[1]])
        center = np.linalg.solve(coef_matrix, dependent_variable)
        point_out_of_image(center)
        return center

    def find_semi_axis(pt, center):
        npt = [(p[0] - center[0], p[1] - center[1]) for p in pt]
        x1, y1, x2, y2, x3, y3 = np.array(npt).flatten()
        coef_matrix = np.array([[x1 ** 2, 2 * x1 * y1, y1 ** 2], [x2 ** 2, 2 * x2 * y2, y2 ** 2], [x3 ** 2, 2 * x3 * y3, y3 ** 2]])
        dependent_variable = np.array([1, 1, 1])
        A, B, C = np.linalg.solve(coef_matrix, dependent_variable)
        if A * C - B ** 2 > 0:
            angle = 0.5 * np.arctan((2 * B) / (A - C))
            axis_coef = np.array([[np.sin(angle) ** 2, np.cos(angle) ** 2], [np.cos(angle) ** 2, np.sin(angle) ** 2]])
            axis_ans = np.array([A, C])
            a, b = np.linalg.solve(axis_coef, axis_ans)
            if a > 0 and b > 0:
                major = 1 / np.sqrt(min(a, b))
                minor = 1 / np.sqrt(max(a, b))
                flattening = (major - minor) / major
                if (major_axis_bound[0] < 2 * major < major_axis_bound[1]) and (
                        minor_axis_bound[0] < 2 * minor < minor_axis_bound[1]) and (flattening < max_flattening):
                    return major, minor, angle
        raise ValueError

    class Accumulator:
        def __init__(self):
            self.accumulator = []

        def get_best_candidate(self):
            self.accumulator = sorted(self.accumulator, key=lambda candidate: candidate[5], reverse=True)
            return self.accumulator[0]

        def evaluate_candidate(self, new_candidate):
            index = self.get_similar_index(new_candidate)
            if index == -1:
                self.add(new_candidate)
            else:
                self.merge(index, new_candidate)

        def add(self, candidate):
            self.accumulator.append(candidate)

        def merge(self, index, candidate):
            score = self.accumulator[index][5] + 1
            self.accumulator[index] = tuple(
                (self.accumulator[index][i] * score + candidate[i]) / (score + 1) if i < 5 else score for i in range(6))

        def get_similar_index(self, new_candidate):
            for idx, candidate in enumerate(self.accumulator):
                if self.is_similar(candidate, new_candidate):
                    return idx
            return -1

        def is_similar(self, candidate, new_candidate):
            center_dist = np.sqrt((candidate[0] - new_candidate[0]) ** 2 + (candidate[1] - new_candidate[1]) ** 2)
            angle_dist = abs(candidate[4] - new_candidate[4])
            angle180 = new_candidate[4] - np.pi if new_candidate[4] >= 0 else new_candidate[4] + np.pi
            angle_dist180 = abs(candidate[4] - angle180)
            angle_final = min(angle_dist, angle_dist180)
            major_axis_dist = abs(max(new_candidate[2], new_candidate[3]) - max(candidate[2], candidate[3]))
            minor_axis_dist = abs(min(new_candidate[2], new_candidate[3]) - min(candidate[2], candidate[3]))
            return (major_axis_dist < similar_major_axis_dist) and (minor_axis_dist < similar_minor_axis_dist) and (
                    center_dist < similar_center_dist) and (angle_final < similar_angle_dist)

    def main_process(img):
        edge = canny_edge_detector(img, canny_sigma, canny_t1, canny_t2)
        edge_pixels = np.array(np.where(edge == 255)).T
        if len(edge_pixels) < 15:
            raise ValueError("No edge!")
        edge_pixels = [p for p in edge_pixels]
        if len(edge_pixels) > 100:
            max_iter = len(edge_pixels) * 20
        accumulator = Accumulator()
        for i in range(max_iter):
            try:
                point_package = randomly_pick_point(edge_pixels)
                center = find_center(edge, point_package)
                semi_major, semi_minor, angle = find_semi_axis(point_package, center)
                candidate = (center[0], center[1], semi_major, semi_minor, angle, 1)
                accumulator.evaluate_candidate(candidate)
            except:
                continue
        return accumulator.get_best_candidate()

    assert len(img.shape) == 2, "This img is not a 2D image"
    assert type(img).__module__ == np.__name__, "This img is not a numpy array"

    best_candidate = main_process(img)
    return best_candidate

# Example usage:
# if __name__ == '__main__':
#     original_image = cv2.imread("heyya.png", 0)
#     best_ellipse = randomized_hough_ellipse_transform(original_image)

#     fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
#     ax[0].set_title("Original")
#     ax[0].imshow(original_image, cmap='gray')
#     ax[0].axis("off")

#     p, q, a, b, angle, score = best_ellipse
#     result_image = original_image.copy()
#     result_image = cv2.ellipse(result_image, (int(p), int(q)), (int(a), int(b)), angle * 180 / np.pi, 0, 360, color=(0, 255, 0), thickness=2)
#     ax[1].set_title("Result")
#     ax[1].imshow(result_image, cmap='gray')
#     ax[1].axis("off")

#     plt.show()
