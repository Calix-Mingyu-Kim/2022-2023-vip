import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Convert RGB to Grayscale
def lightness_method(img):
    grayImage = np.zeros(img.shape)
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])

    grayImage = img.copy()

    for i in range(3):
        grayscale = (np.minimum(r,g,b)+np.maximum(r,g,b))/2
        grayImage[:,:,i] = grayscale
        
    return grayImage  
    
def avg_method(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = (r+g+b)/3
    # grayImage = np.zeros(img.shape)
    grayImage = img.copy()

    for i in range(3):
        grayImage[:,:,i] = grayscale
        
    return grayImage  

def luminosity(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = 0.2126 * r + 0.7152* g + 0.0722 * b 
    grayImage = img.copy()

    for i in range(3):
        grayImage[:,:,i] = grayscale
        
    return grayImage

image = mpimg.imread("bdd100k_sample/7d2f7975-e0c1c5a7.jpg")

originalImg = image
lightGrayImg = lightness_method(image)
avgGrayImg = avg_method(image)  
lumGrayImg = luminosity(image)

# images = [originalImg, lightGrayImg, avgGrayImg, lumGrayImg]
    
# for i in range(len(images)):
#     plt.subplot(5,5,i+1)
#     # image = plt.imread(images[i])
#     plt.imshow(images[i])
# plt.imshow(image)
# plt.imshow(lightGrayImg)
# plt.imshow(avgGrayImg)
plt.imshow(lumGrayImg)
plt.show()
