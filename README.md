# 🖼️ Image Panorama Stitcher

A Python-based panorama stitching pipeline that takes three overlapping images and merges them into a single seamless wide-angle panorama using SIFT feature detection and homography-based alignment.

---

## How It Works

The pipeline follows these steps:

1. **Feature Detection** — SIFT (Scale-Invariant Feature Transform) is used to detect and describe keypoints in each image.
2. **Feature Matching** — A Brute-Force Matcher with Lowe's ratio test (threshold: 0.75) filters for high-quality matches between adjacent image pairs (Image 1↔2 and Image 2↔3).
3. **Homography Estimation** — RANSAC-based homography matrices (`H12`, `H23`) are computed to find the geometric transformation that maps each side image onto the coordinate space of the center image.
4. **Canvas Warping** — All three images are warped onto a shared canvas, with Image 2 placed at the center as the reference.
5. **Naive Stitching** — A simple overlay (last-write-wins) produces an initial rough panorama.
6. **Alpha Blending** — Overlapping regions are blended with a 50/50 average to reduce visible seams.
7. **Cropping** — Black border regions are automatically detected and cropped from the final output.

---

## Requirements

- Python 3.7+
- OpenCV (with contrib for SIFT support)
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install opencv-contrib-python numpy matplotlib
```

> ⚠️ SIFT is available in `opencv-contrib-python`. The standard `opencv-python` package does **not** include it.

---

## Usage

1. Place your three overlapping images in the same directory as the script and name them:
   - `image1.jpeg` — left image
   - `image2.jpeg` — center image (used as reference)
   - `image3.jpeg` — right image

2. Run the script:

```bash
python panorama.py
```

3. Output files will be saved to the same directory:
   - `naive_panorama.jpg` — simple overlay stitch
   - `final_panorama.jpg` — blended and cropped panorama

---

## Output

The script displays intermediate visualizations at each stage:

| Stage | Description |
|---|---|
| SIFT Keypoints | Detected keypoints overlaid on each image |
| Feature Matches | Matched keypoint pairs between adjacent images |
| Naive Stitch | Raw warped panorama before blending |
| Final Panorama | Blended and cropped result |

---

## Tips for Best Results

- Images should have **30–50% overlap** between adjacent frames.
- Capture images by rotating around a fixed point (e.g., tripod) to minimize parallax error.
- Consistent lighting across all three images reduces visible seams after blending.
- For better blending quality, consider replacing the 50/50 alpha blend with gradient/feather blending or multi-band blending.

---

## Limitations

- Currently supports exactly **3 images** as input.
- The simple alpha blend may still show seams under significant exposure differences.
- Large perspective distortions (non-planar scenes) may cause alignment artifacts.
