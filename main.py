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

# Converting BGR to RGB 
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

# Display images
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(img1_rgb)
plt.title("Image 1")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img2_rgb)
plt.title("Image 2")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img3_rgb)
plt.title("Image 3")
plt.axis("off")

plt.show()
