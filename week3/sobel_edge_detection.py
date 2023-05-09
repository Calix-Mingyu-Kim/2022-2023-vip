import numpy as np

#Sobel Edge detection
def sobel_edge(img):
    h, w = img.shape
    
    # initialize x, y direction kernels 
    x_direction = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=object)
    y_direction = [[-1,-2,-1],[0,0,0],[1,2,1]]
    
    # initialize output size
    sobel_G = img.copy()
    # Gx = img.copy()
    # Gy = img.copy()
    
    # convolution 
    for row in range(0, h-2):
        for column in range(0, w-2):
            input = img[row:row+3, column:column+3]

            # find Gx and Gy using x, y direction kernels and the 3x3 input 'patch'
            Gx = sum(map(sum, np.multiply(input, x_direction)))
            Gy = sum(map(sum, np.multiply(input, y_direction)))
            
            # image graditude of gradient G
            sobel_G[row+1, column+1] = (Gx ** 2 + Gy ** 2) ** 0.5
                
    return sobel_G