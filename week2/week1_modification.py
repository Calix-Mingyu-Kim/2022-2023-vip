import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Convert RGB to Grayscale
def lightness_method(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])

    grayscale = (np.minimum(r,g,b)+np.maximum(r,g,b))/2
        
    return grayscale  
    
def avg_method(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = (r+g+b)/3
    # grayImage = np.zeros(img.shape)
    # grayImage = img.copy()

    # for i in range(3):
    #     grayImage[:,:,i] = grayscale
        
    return grayscale  

def luminosity(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = 0.2126 * r + 0.7152* g + 0.0722 * b 
        
    return grayscale

image = mpimg.imread("bdd100k_sample/7d2f7975-e0c1c5a7.jpg")

originalImg = image
lightGrayImg = lightness_method(image)
avgGrayImg = avg_method(image)  
lumGrayImg = luminosity(image)

# # plt.imshow(image)
# plt.imshow(lightGrayImg, cmap='gray')
# # plt.imshow(avgGrayImg, cmap='gray')
# # plt.imshow(lumGrayImg, cmap='gray')
# print(lumGrayImg.shape)
# plt.show()

# subplot(r,c) provide the no. of rows and columns
fig, axs = plt.subplots(2,2) 

# Show each image in one window
axs[0,0].imshow(image)
axs[0,1].imshow(lightGrayImg, cmap='gray')
axs[1,0].imshow(avgGrayImg, cmap='gray')
axs[1,1].imshow(lumGrayImg, cmap='gray')

plt.show()