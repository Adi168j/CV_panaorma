import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


print("OpenCV Version:", cv2.__version__)

try:
    sift = cv2.SIFT_create()
    print("SIFT is working correctly ")
except:
    print("SIFT is NOT working ")





img1 = cv2.imread("image1.jpeg")
img2 = cv2.imread("image2.jpeg")
img3 = cv2.imread("image3.jpeg")


if img1 is None or img2 is None or img3 is None:
    print("Error loading images. Check file paths.")
    exit()

# # Converting BGR to RGB 
# img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

# # Display images
# plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.imshow(img1_rgb)
# plt.title("Image 1")
# plt.axis("off")

# plt.subplot(1,3,2)
# plt.imshow(img2_rgb)
# plt.title("Image 2")
# plt.axis("off")

# plt.subplot(1,3,3)
# plt.imshow(img3_rgb)
# plt.title("Image 3")
# plt.axis("off")

# plt.show()



gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# Create SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and descriptors
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


# Convert to RGB for matplotlib display
img1_kp = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
img2_kp = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)
img3_kp = cv2.cvtColor(img3_kp, cv2.COLOR_BGR2RGB)


# Display keypoints
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
plt.pause(3)


bf = cv2.BFMatcher()

# Match descriptors (Image 1 and Image 2)
matches12 = bf.knnMatch(des1, des2, k=2)

# Match descriptors (Image 2 and Image 3)
matches23 = bf.knnMatch(des2, des3, k=2)


# Apply Lowe's Ratio Test
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


# Draw matches
match_img12 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches12, None, flags=2)
match_img23 = cv2.drawMatches(img2, kp2, img3, kp3, good_matches23, None, flags=2)


# Convert to RGB for display
match_img12 = cv2.cvtColor(match_img12, cv2.COLOR_BGR2RGB)
match_img23 = cv2.cvtColor(match_img23, cv2.COLOR_BGR2RGB)


# Display matches
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


