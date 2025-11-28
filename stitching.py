#!/usr/bin/env python3

from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

FEATURE_MAX_SIZE = 1600
SIFT_NFEATURES = 1500

DEBUG_OUT_DIR: str | None = None

class Image:
    def __init__(self, path: str, size: int | None = None) -> None:
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image {path}")
        if size:
            h, w = self.image.shape[:2]
            if max(h, w) > size:
                if w > h:
                    self.image = cv2.resize(self.image, (size, int(h * size / w)))
                else:
                    self.image = cv2.resize(self.image, (int(w * size / h), size))
        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3, dtype=np.float64)
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)
        self._scale_for_features = 1.0

    def compute_features(self, feature_max_size: int = FEATURE_MAX_SIZE, nfeatures: int = SIFT_NFEATURES) -> None:
        h, w = self.image.shape[:2]
        scale = 1.0
        if max(h, w) > feature_max_size:
            scale = feature_max_size / float(max(h, w))
            small = cv2.resize(self.image, (int(w * scale), int(h * scale)))
        else:
            small = self.image.copy()
        self._scale_for_features = scale
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        kps_small, desc = sift.detectAndCompute(gray_small, None)
        if kps_small is None or desc is None:
            self.keypoints = []
            self.features = None
            return
        if scale != 1.0:
            kps = []
            for kp in kps_small:
                kps.append(cv2.KeyPoint(kp.pt[0] / scale, kp.pt[1] / scale, kp.size / scale, kp.angle, kp.response, kp.octave, kp.class_id))
            self.keypoints = kps
        else:
            self.keypoints = kps_small
        self.features = desc.astype(np.float32)

class PairMatch:
    def __init__(self, img_a: Image, img_b: Image, matches: list | None = None) -> None:
        self.image_a = img_a
        self.image_b = img_b
        self.matches = matches or []
        self.H: np.ndarray | None = None
        self.status = None
        self.overlap = None
        self.area_overlap = None

    def compute_homography(self) -> None:
        if not self.matches or len(self.matches) < 3:
            self.H = None
            self.status = None
            return
        pts_a, pts_b = [], []
        for m in self.matches:
            pts_a.append(self.image_a.keypoints[m.queryIdx].pt)
            pts_b.append(self.image_b.keypoints[m.trainIdx].pt)
        if len(pts_a) < 3 or len(pts_b) < 3:
            self.H = None
            return
        pts_a, pts_b = np.float32(pts_a), np.float32(pts_b)
        affine_2x3, mask = cv2.estimateAffinePartial2D(pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if affine_2x3 is None:
            self.H = None
            return
        H3 = np.eye(3)
        H3[0:2, 0:3] = affine_2x3
        self.H = H3
        self.status = mask.ravel().astype(np.uint8) if mask is not None else np.ones(len(pts_a), dtype=np.uint8)

def single_weights_matrix(shape: Tuple[int, int, ...]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    return dist / dist.max() if dist.max() > 0 else np.ones_like(dist, dtype=np.float32)

def get_new_parameters(base: np.ndarray | None, image: np.ndarray, H: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
    def corners_of(img: np.ndarray):
        h, w = img.shape[:2]
        return np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    
    img_corners = corners_of(image)
    img_corners_h = np.hstack([img_corners, np.ones((4,1))])
    warped = (H @ img_corners_h.T).T
    warped = warped[:, :2] / warped[:, 2:3]

    if base is None:
        min_x, min_y = warped.min(axis=0)
        max_x, max_y = warped.max(axis=0)
        tx, ty = -min_x if min_x<0 else 0, -min_y if min_y<0 else 0
        added_offset = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
        return (int(np.ceil(max_x+tx)), int(np.ceil(max_y+ty))), added_offset

    base_h, base_w = base.shape[:2]
    base_corners = np.array([[0,0],[base_w,0],[base_w,base_h],[0,base_h]], dtype=np.float32)
    all_pts = np.vstack([base_corners, warped])
    min_xy, max_xy = all_pts.min(axis=0), all_pts.max(axis=0)
    tx, ty = -min_xy[0] if min_xy[0]<0 else 0, -min_xy[1] if min_xy[1]<0 else 0
    added_offset = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    return (int(np.ceil(max_xy[0]+tx)), int(np.ceil(max_xy[1]+ty))), added_offset


def add_image(panorama: np.ndarray | None, image: Image, offset: np.ndarray, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Ensure offset and image.H are float64
    offset = offset.astype(np.float64)
    image_H = image.H.astype(np.float64)
    H = offset @ image_H

    size, added_offset = get_new_parameters(panorama, image.image, H)
    composed_T = added_offset @ H

    new_image = cv2.warpPerspective(image.image, composed_T, size)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros_like(new_image[:, :, 0])
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size)
        weights = cv2.warpPerspective(weights, added_offset, size)

    # Compute image weights safely
    weight_2d = cv2.warpPerspective(single_weights_matrix(image.image.shape), composed_T, size)
    image_weights = np.repeat(weight_2d[:, :, np.newaxis], 3, axis=2)

    denom = weights + image_weights[:, :, 0]
    normalized_weights = np.zeros_like(weights, dtype=np.float32)
    mask = denom != 0
    normalized_weights[mask] = (weights[mask] / denom[mask])

    combined = np.where(
        np.repeat(weights[:, :, np.newaxis]==0, 3, axis=2),
        new_image,
        (new_image*(1-normalized_weights[:,:,np.newaxis]) + panorama*normalized_weights[:,:,np.newaxis])
    )

    panorama = combined.astype(np.uint8)
    weights = denom
    if weights.max() > 0:
        weights = weights / weights.max()

    return panorama, added_offset @ offset, weights


def simple_blending(images: List[Image]) -> np.ndarray:
    global DEBUG_OUT_DIR
    panorama = None
    weights = None
    offset = np.eye(3)
    panorama_mask = None
    step = 0
    for img in tqdm(images, desc="Blending", unit="img"):
        H = offset @ img.H
        size, added_offset = get_new_parameters(panorama, img.image, H)
        composed_T = added_offset @ H
        ones_mask_src = np.ones_like(img.image[:, :, 0], dtype=np.uint8)
        warped_new_mask = cv2.warpPerspective(ones_mask_src, composed_T, size)
        warped_new_mask = (warped_new_mask>0).astype(np.uint8)
        prev_mask_warped = np.zeros_like(warped_new_mask) if panorama_mask is None else cv2.warpPerspective(
        panorama_mask.astype(np.uint8),
        added_offset.astype(np.float64),  # <- cast to float
        size)

        panorama, offset, weights = add_image(panorama, img, offset, weights)
        step_mask = warped_new_mask.copy()
        if DEBUG_OUT_DIR:
            os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_OUT_DIR,f"stitched_step_{step+1:02d}.png"), panorama)
            cv2.imwrite(os.path.join(DEBUG_OUT_DIR,f"mask_step_{step+1:02d}.png"), step_mask*255)
        panorama_mask = (prev_mask_warped | warped_new_mask).astype(np.uint8) if panorama_mask is not None else warped_new_mask
        step += 1
    return panorama

def main():
    global DEBUG_OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--size", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO if args["verbose"] else logging.WARNING)

    data_dir = args["data_dir"]
    valid_exts = {".jpg", ".png", ".bmp", ".jpeg"}
    image_paths = sorted([str(p) for p in data_dir.iterdir() if p.suffix.lower() in valid_exts])
    images = [Image(p, args.get("size")) for p in image_paths]
    DEBUG_OUT_DIR = str(data_dir / "results" / "debug_masks")
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

    for img in tqdm(images, desc="Extracting features"):
        img.compute_features()

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))

    if images:
        images[0].H = np.eye(3)
    for i in range(1, len(images)):
        prev, curr = images[i-1], images[i]
        if prev.features is None or curr.features is None:
            curr.H = prev.H.copy()
            continue
        raw_matches = flann.knnMatch(prev.features, curr.features, k=2)
        good = [m for m_n in raw_matches if len(m_n)==2 and m_n[0].distance<0.7*m_n[1].distance for m in [m_n[0]]]
        pm = PairMatch(prev, curr, good)
        pm.compute_homography()
        curr.H = prev.H @ pm.H if pm.H is not None else prev.H.copy()

    pano = simple_blending(images)
    out_dir = data_dir / "results"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(str(out_dir / "panorama.jpg"), pano)
    logging.info("Saved final panorama at %s", str(out_dir / "panorama.jpg"))

if __name__=="__main__":
    main()
