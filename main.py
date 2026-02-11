import cv2
import matplotlib.pyplot as plt
import os



def load_images(image_paths):
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        
        if img is None:
            print(f"Error loading image: {path}")
        else:
            print(f"Loaded image: {path}")
            images.append(img)
    
    return images




def display_images(images, titles=None):
    plt.figure(figsize=(15, 5))
    
    for i, img in enumerate(images):
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, len(images), i+1)
        plt.imshow(img_rgb)
        
        if titles:
            plt.title(titles[i])
        
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    image_paths = ["image1.jpeg", "image2.jpeg", "image3.jpeg"]
    
    images = load_images(image_paths)
    
    display_images(images, titles=["Image 1", "Image 2", "Image 3"])
