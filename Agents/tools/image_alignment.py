#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _failure_meta(
    pre_path: Path,
    post_path: Path,
    out_dir: Path,
    *,
    reason: str,
    method: str = "none",
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    meta = {
        "applied": False,
        "alignment_failed": True,
        "method": method,
        "reason": reason,
        "details": details or {},
        "aligned_pre_path": str(pre_path),
        "aligned_post_path": str(post_path),
    }
    _save_json(out_dir / "alignment_meta.json", meta)
    return meta


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _quality_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _compute_reprojection_error(src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray, mask: np.ndarray | None) -> float:
    if H is None or src_pts.size == 0 or dst_pts.size == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(src_pts, H)
    errors = np.linalg.norm(projected - dst_pts, axis=2).reshape(-1)
    if mask is not None:
        keep = mask.ravel().astype(bool)
        if keep.any():
            errors = errors[keep]
    if errors.size == 0:
        return float("inf")
    return float(np.mean(errors))


def _compute_orb_alignment_score(
    *,
    inlier_ratio: float,
    reprojection_error: float,
    bbox_area_ratio: float,
) -> float:
    # Heuristic confidence used for tool-level reliability assessment.
    ratio_score = _clamp01(inlier_ratio / 0.5)
    reproj_score = _clamp01(1.0 - (reprojection_error / 8.0))
    bbox_score = _clamp01(bbox_area_ratio / 0.3)
    return float(0.5 * ratio_score + 0.3 * reproj_score + 0.2 * bbox_score)


def _template_match_crop(
    pre_path: Path,
    post_path: Path,
    out_dir: Path,
    *,
    min_scale: float = 0.35,
    max_scale: float = 1.15,
    steps: int = 24,
    min_confidence: float = 0.20,
    fallback_reason: str = "",
) -> Dict[str, Any]:
    _ensure_dir(out_dir)
    pre = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    post = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if pre is None or post is None:
        return _failure_meta(pre_path, post_path, out_dir, reason="image_read_failed")

    pre_h, pre_w = pre.shape[:2]
    post_h, post_w = post.shape[:2]
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    best = {"score": -1.0, "scale": 1.0, "x": 0, "y": 0, "w": min(post_w, pre_w), "h": min(post_h, pre_h)}
    for s in np.linspace(min_scale, max_scale, steps):
        tw = max(24, int(post_w * float(s)))
        th = max(24, int(post_h * float(s)))
        if tw >= pre_w or th >= pre_h:
            continue
        tpl = cv2.resize(post_gray, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(pre_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if float(max_val) > float(best["score"]):
            best = {"score": float(max_val), "scale": float(s), "x": int(max_loc[0]), "y": int(max_loc[1]), "w": int(tw), "h": int(th)}

    x1 = int(best["x"])
    y1 = int(best["y"])
    x2 = min(pre_w, x1 + int(best["w"]))
    y2 = min(pre_h, y1 + int(best["h"]))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return _failure_meta(
            pre_path,
            post_path,
            out_dir,
            reason="invalid_bbox",
            method="template_matching_fallback",
            details={"best_match": best},
        )

    confidence = float(best["score"])
    if confidence < float(min_confidence):
        return _failure_meta(
            pre_path,
            post_path,
            out_dir,
            reason="template_match_low_confidence",
            method="template_matching_fallback",
            details={
                "best_match": best,
                "min_confidence": min_confidence,
                "fallback_reason": fallback_reason,
            },
        )

    pre_crop = pre[y1:y2, x1:x2]
    pre_crop_path = out_dir / "pre_aligned_crop.png"
    cv2.imwrite(str(pre_crop_path), pre_crop)
    post_resized = cv2.resize(post, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    post_resized_path = out_dir / "post_resized_to_pre_crop.png"
    cv2.imwrite(str(post_resized_path), post_resized)
    vis = pre.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    vis_path = out_dir / "pre_match_bbox_vis.png"
    cv2.imwrite(str(vis_path), vis)

    meta = {
        "applied": True,
        "alignment_failed": False,
        "method": "template_matching_fallback",
        "fallback_reason": fallback_reason,
        "pre_size": {"w": pre_w, "h": pre_h},
        "post_size": {"w": post_w, "h": post_h},
        "search_config": {
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "steps": int(steps),
            "match_metric": "TM_CCOEFF_NORMED",
            "min_confidence": float(min_confidence),
            "grayscale": True,
        },
        "best_match": best,
        "alignment_confidence": confidence,
        "alignment_quality": _quality_label(confidence),
        "bbox_xyxy": [x1, y1, x2, y2],
        "aligned_pre_path": str(pre_crop_path),
        "aligned_post_path": str(post_resized_path),
        "vis_path": str(vis_path),
    }
    _save_json(out_dir / "alignment_meta.json", meta)
    return meta


def match_and_crop_pre_to_post(pre_path: Path, post_path: Path, out_dir: Path) -> Dict[str, Any]:
    """
    details：ORB details + RANSAC details，details。
    """
    _ensure_dir(out_dir)
    pre = cv2.imread(str(pre_path), cv2.IMREAD_COLOR)
    post = cv2.imread(str(post_path), cv2.IMREAD_COLOR)
    if pre is None or post is None:
        return _failure_meta(pre_path, post_path, out_dir, reason="image_read_failed")

    pre_h, pre_w = pre.shape[:2]
    post_h, post_w = post.shape[:2]
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    min_keypoints = 8
    min_good_matches = 10
    min_inliers = 8
    min_inlier_ratio = 0.25
    max_reprojection_error = 8.0

    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    kp_pre, des_pre = orb.detectAndCompute(pre_gray, None)
    kp_post, des_post = orb.detectAndCompute(post_gray, None)
    if des_pre is None or des_post is None or len(kp_pre) < min_keypoints or len(kp_post) < min_keypoints:
        return _template_match_crop(
            pre_path,
            post_path,
            out_dir,
            fallback_reason="insufficient_orb_keypoints_or_descriptors",
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_post, des_pre, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_good_matches:
        return _template_match_crop(
            pre_path,
            post_path,
            out_dir,
            fallback_reason="insufficient_good_matches",
        )

    src_pts = np.float32([kp_post[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_pre[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return _template_match_crop(pre_path, post_path, out_dir, fallback_reason="homography_estimation_failed")
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    if inliers < min_inliers:
        return _template_match_crop(pre_path, post_path, out_dir, fallback_reason="insufficient_inliers")

    inlier_ratio = float(inliers) / float(len(good)) if good else 0.0
    reprojection_error = _compute_reprojection_error(src_pts, dst_pts, H, mask)

    post_corners = np.float32([[0, 0], [post_w - 1, 0], [post_w - 1, post_h - 1], [0, post_h - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(post_corners, H).reshape(-1, 2)
    xs = warped[:, 0]
    ys = warped[:, 1]
    x1 = max(0, int(np.floor(xs.min())))
    y1 = max(0, int(np.floor(ys.min())))
    x2 = min(pre_w, int(np.ceil(xs.max())))
    y2 = min(pre_h, int(np.ceil(ys.max())))
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return _template_match_crop(pre_path, post_path, out_dir, fallback_reason="invalid_projected_bbox")

    bbox_area_ratio = float((x2 - x1) * (y2 - y1)) / float(pre_w * pre_h)
    reliability_checks = {
        "min_inliers_passed": inliers >= min_inliers,
        "min_inlier_ratio_passed": inlier_ratio >= min_inlier_ratio,
        "max_reprojection_error_passed": reprojection_error <= max_reprojection_error,
        "valid_bbox_passed": x2 > x1 + 4 and y2 > y1 + 4,
    }
    if not all(reliability_checks.values()):
        return _template_match_crop(
            pre_path,
            post_path,
            out_dir,
            fallback_reason="orb_reliability_check_failed",
        )

    alignment_confidence = _compute_orb_alignment_score(
        inlier_ratio=inlier_ratio,
        reprojection_error=reprojection_error,
        bbox_area_ratio=bbox_area_ratio,
    )

    pre_crop = pre[y1:y2, x1:x2]
    pre_crop_path = out_dir / "pre_aligned_crop.png"
    cv2.imwrite(str(pre_crop_path), pre_crop)
    post_resized = cv2.resize(post, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
    post_resized_path = out_dir / "post_resized_to_pre_crop.png"
    cv2.imwrite(str(post_resized_path), post_resized)
    vis = pre.copy()
    cv2.polylines(vis, [np.int32(warped)], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    vis_path = out_dir / "pre_match_bbox_vis.png"
    cv2.imwrite(str(vis_path), vis)

    meta = {
        "applied": True,
        "alignment_failed": False,
        "method": "orb_ransac_homography",
        "pre_size": {"w": pre_w, "h": pre_h},
        "post_size": {"w": post_w, "h": post_h},
        "orb": {
            "kp_pre": len(kp_pre),
            "kp_post": len(kp_post),
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "reprojection_error": reprojection_error,
            "ratio_test_threshold": 0.75,
            "ransac_reprojection_threshold": 5.0,
            "reliability_thresholds": {
                "min_keypoints": min_keypoints,
                "min_good_matches": min_good_matches,
                "min_inliers": min_inliers,
                "min_inlier_ratio": min_inlier_ratio,
                "max_reprojection_error": max_reprojection_error,
            },
            "reliability_checks": reliability_checks,
        },
        "alignment_confidence": alignment_confidence,
        "alignment_quality": _quality_label(alignment_confidence),
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_xyxy": [x1, y1, x2, y2],
        "warped_corners_xy": warped.tolist(),
        "aligned_pre_path": str(pre_crop_path),
        "aligned_post_path": str(post_resized_path),
        "vis_path": str(vis_path),
    }
    _save_json(out_dir / "alignment_meta.json", meta)
    return meta


