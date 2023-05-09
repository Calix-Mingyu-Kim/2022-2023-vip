import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Convert RGB to Grayscale
def luminosity(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = 0.2126 * r + 0.7152* g + 0.0722 * b 
        
    return grayscale

# Average filter
def avg_filter(img):
    # initialize kernel [1/9] 3x3
    kernel = np.ones((3, 3), np.float32) / 9
    h, w = img.shape
    
    # initialize output size
    output = np.zeros((h-2,w-2))
    
    # convolution 
    for row in range(0, img.shape[0]-2):
        for column in range(0, img.shape[1]-2):
            input = img[row:row+3, column:column+3]
            # multiply input and kernel for each 'patch'
            output[row,column] = sum(map(sum, np.multiply(input, kernel)))

    return output

def median_filter(img):
    h, w = img.shape
    
    # initialize output size
    output = img.copy()
    
    # convolution 
    for row in range(0, h-2):
        for column in range(0, w-2):
            input = img[row:row+3, column:column+3]
            # find median of each 'patch' and apply to output
            output[row+1, column+1] = np.median(input)

    return output
    
            
            
image = mpimg.imread("bdd100k_sample/7d4a9094-0455564b.jpg")

lumGrayImg = luminosity(image)
avgFilterImg = avg_filter(lumGrayImg)
medianFilterImg = median_filter(lumGrayImg)

#subplot(r,c) provide the no. of rows and columns
fig, axs = plt.subplots(1,2) 

# Show each image in one window
axs[0].imshow(lumGrayImg, cmap='gray')
axs[0].title.set_text('Original grayscale image')
axs[1].imshow(medianFilterImg, cmap='gray')
axs[1].title.set_text('Median Filter')

# axs[1].imshow(avgFilterImg, cmap='gray')
# axs[1].title.set_text('Average Filter')
# axs[2].imshow(medianFilterImg, cmap='gray')
# axs[2].title.set_text('Median Filter')

plt.show()


# plt.imshow(lumGrayImg, cmap='gray')
# print(lumGrayImg[0][0])
# print(avgFilterImg[0][0])
# plt.show()