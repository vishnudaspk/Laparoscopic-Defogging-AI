import cv2
import numpy as np


def compute_dc(im, sz=15):
    b, g, r = cv2.split(im)
    min_dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(min_dc, kernel)
    return dark, min_dc


def apply_guided_filter(im, p, r=15, eps=0.0001):
    mean_i = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def refine_dc(dc, min_dc, r=15, eps=0.0001):
    """Apply guided filter to refine the dark channel."""
    dc_rfd = apply_guided_filter(min_dc, dc, r, eps)
    return dc_rfd


def get_refined_dc(input_img, sz=15, r=15, eps=0.0001):
    dc, min_dc = compute_dc(input_img.astype('float64') / 255, sz)
    dc = refine_dc(dc, min_dc, r, eps) * 255
    dc[dc < 0] = 0
    dc[dc > 255] = 255
    return np.uint8(dc), input_img


def add_guide_channel(input_img, sz=15, r=15, eps=0.0001, bypass=False):
    """Compute and concatenate the dark channel prior as a 4th channel.

    Args:
        input_img: RGB uint8 image (H, W, 3)
        sz:        Dark channel erosion kernel size (odd int)
        r:         Guided filter radius
        eps:       Guided filter regularization
        bypass:    If True, return a zero-valued 4th channel (skip computation)

    Returns:
        RGBD: (H, W, 4) uint8 array
        dc_vis: (H, W) uint8 dark channel visualization (or zeros if bypass)
    """
    if bypass:
        h, w = input_img.shape[:2]
        dark = np.zeros((h, w, 1), dtype=np.uint8)
        rgb_d = np.concatenate((input_img, dark), axis=2)
        return rgb_d, np.zeros((h, w), dtype=np.uint8)

    dc, min_dc = compute_dc(input_img.astype('float64') / 255, sz)
    dc_rfd = refine_dc(dc, min_dc, r, eps) * 255
    dc_rfd[dc_rfd < 0] = 0
    dc_rfd[dc_rfd > 255] = 255
    dc_vis = np.uint8(dc_rfd)
    dark = np.expand_dims(dc_vis, axis=2)
    rgb_d = np.concatenate((input_img, dark), axis=2)
    return rgb_d, dc_vis
