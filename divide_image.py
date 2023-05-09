import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("/Users/calixkim/VIP27920/bdd100k/images/100k/val/b1cac6a7-04e33135.jpg")


def img_to_grid(img,row,col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]),row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]),col)]
    grid = [img[j:jj,i:ii,:] for j,jj in ww for i,ii in hh]
    return grid, len(ww), len(hh)

def plot_grid(grid,row,col,h=5,w=5):
    fig, ax = plt.subplots(nrows=row, ncols=col)
    [axi.set_axis_off() for axi in ax.ravel()]

    fig.set_figheight(h)
    fig.set_figwidth(w)
    c = 0
    for row in ax:
        for col in row:
            col.imshow(np.flip(grid[c],axis=-1))
            c+=1
    plt.show()

if __name__=='__main__':
    row, col = 13, 13
    grid , r,c = img_to_grid(img,row,col)
    plot_grid(grid,r,c)
        
