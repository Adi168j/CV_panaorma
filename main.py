import cv2
import matplotlib.pyplot as plt
import numpy as np


# CHECK SIFT


print("OpenCV Version:", cv2.__version__)

try:
    sift = cv2.SIFT_create()
    print("SIFT is working correctly ")
except:
    print("SIFT is NOT working ")


# LOAD IMAGES


img1 = cv2.imread("image1.jpeg")
img2 = cv2.imread("image2.jpeg")
img3 = cv2.imread("image3.jpeg")

if img1 is None or img2 is None or img3 is None:
    print("Error loading images. Check file paths.")
    exit()


# CONVERSION TO GRAYSCALE


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# SIFT FEATURE DETECTION

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
kp3, des3 = sift.detectAndCompute(gray3, None)

print("Keypoints in Image 1:", len(kp1))
print("Keypoints in Image 2:", len(kp2))
print("Keypoints in Image 3:", len(kp3))


# Draw keypoints on images
img1_kp = cv2.drawKeypoints(img1, kp1, None)
img2_kp = cv2.drawKeypoints(img2, kp2, None)
img3_kp = cv2.drawKeypoints(img3, kp3, None)

img1_kp = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
img2_kp = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)
img3_kp = cv2.cvtColor(img3_kp, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img1_kp)
plt.title("SIFT Keypoints - Image 1")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img2_kp)
plt.title("SIFT Keypoints - Image 2")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img3_kp)
plt.title("SIFT Keypoints - Image 3")
plt.axis("off")

plt.show(block=False)


# FEATURE MATCHING


bf = cv2.BFMatcher()

matches12 = bf.knnMatch(des1, des2, k=2)
matches23 = bf.knnMatch(des2, des3, k=2)

good_matches12 = []
for m, n in matches12:
    if m.distance < 0.75 * n.distance:
        good_matches12.append(m)

good_matches23 = []
for m, n in matches23:
    if m.distance < 0.75 * n.distance:
        good_matches23.append(m)

print("Good Matches (Image1-Image2):", len(good_matches12))
print("Good Matches (Image2-Image3):", len(good_matches23))


match_img12 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches12, None, flags=2)
match_img23 = cv2.drawMatches(img2, kp2, img3, kp3, good_matches23, None, flags=2)

match_img12 = cv2.cvtColor(match_img12, cv2.COLOR_BGR2RGB)
match_img23 = cv2.cvtColor(match_img23, cv2.COLOR_BGR2RGB)


plt.figure(figsize=(15,8))

plt.subplot(2,1,1)
plt.imshow(match_img12)
plt.title("Feature Matches: Image 1 ↔ Image 2")
plt.axis("off")

plt.subplot(2,1,2)
plt.imshow(match_img23)
plt.title("Feature Matches: Image 2 ↔ Image 3")
plt.axis("off")

plt.show(block=False)
plt.pause(3)


# HOMOGRAPHY 


src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches12]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches12]).reshape(-1,1,2)

H12, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print("Homography Matrix H12:")
print(H12)

src_pts_23 = np.float32([kp3[m.trainIdx].pt for m in good_matches23]).reshape(-1,1,2)
dst_pts_23 = np.float32([kp2[m.queryIdx].pt for m in good_matches23]).reshape(-1,1,2)

H23, mask23 = cv2.findHomography(src_pts_23, dst_pts_23, cv2.RANSAC, 5.0)

print("Homography Matrix H23:")
print(H23)


# CREATE CANVAS


h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]

canvas_height = max(h1, h2, h3) * 2
canvas_width = (w1 + w2 + w3)

canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

center_x = w1
center_y = canvas_height // 4

canvas[center_y:center_y+h2, center_x:center_x+w2] = img2



# WARP IMAGES


warp_img1 = cv2.warpPerspective(
    img1,
    np.array([[1,0,center_x],[0,1,center_y],[0,0,1]]) @ H12,
    (canvas_width, canvas_height)
)

warp_img3 = cv2.warpPerspective(
    img3,
    np.array([[1,0,center_x],[0,1,center_y],[0,0,1]]) @ H23,
    (canvas_width, canvas_height)
)


# NAIVE STITCH (UNCHANGED)


naive_panorama = canvas.copy()

mask1 = warp_img1 > 0
naive_panorama[mask1] = warp_img1[mask1]

mask3 = warp_img3 > 0
naive_panorama[mask3] = warp_img3[mask3]


plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(naive_panorama, cv2.COLOR_BGR2RGB))
plt.title("Naive Stitch")
plt.axis("off")
plt.show(block=False)
plt.pause(3)

cv2.imwrite("naive_panorama.jpg", naive_panorama)


#  BLENDING 


blended_panorama = canvas.copy()

# Blend warp_img1
overlap1 = (warp_img1 > 0) & (blended_panorama > 0)
only1 = (warp_img1 > 0) & (blended_panorama == 0)

blended_panorama[overlap1] = (
    0.5 * warp_img1[overlap1] +
    0.5 * blended_panorama[overlap1]
)

blended_panorama[only1] = warp_img1[only1]

# Blend warp_img3
overlap3 = (warp_img3 > 0) & (blended_panorama > 0)
only3 = (warp_img3 > 0) & (blended_panorama == 0)

blended_panorama[overlap3] = (
    0.5 * warp_img3[overlap3] +
    0.5 * blended_panorama[overlap3]
)

blended_panorama[only3] = warp_img3[only3]

blended_panorama = blended_panorama.astype(np.uint8)



# CROP BLACK REGIONS


gray = cv2.cvtColor(blended_panorama, cv2.COLOR_BGR2GRAY)
coords = np.column_stack(np.where(gray > 0))

y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

final_panorama = blended_panorama[y_min:y_max, x_min:x_max]


plt.figure(figsize=(12,6))
plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
plt.title("Final Cropped Panorama")
plt.axis("off")
plt.show(block=False)
plt.pause(3)

cv2.imwrite("final_panorama.jpg", final_panorama)
