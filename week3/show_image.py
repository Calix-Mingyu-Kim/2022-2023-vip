import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sobel_edge_detection import sobel_edge

#Convert RGB to Grayscale
def luminosity(img):
    r, g, b = np.array(img[:, :, 0]), np.array(img[:, :, 1]), np.array(img[:, :, 2])
    
    grayscale = 0.2126 * r + 0.7152* g + 0.0722 * b 
        
    return grayscale
            
image = luminosity(mpimg.imread("bdd100k_sample/7d4a9094-0455564b.jpg"))

sobel_edge_image = sobel_edge(image)

#subplot(r,c) provide the no. of rows and columns
# fig, axs = plt.subplots(1,2) 

# Show each image in one window (Gx, Gy)
# axs[0].imshow(x, cmap='gray')
# axs[0].title.set_text('Gx')
# axs[1].imshow(y, cmap='gray')
# axs[1].title.set_text('Gy')

plt.imshow(sobel_edge_image, cmap='gray')
plt.title("Sobel edge detection")
plt.show()