import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy import misc
import math
import sys 

#讀取照片
img_1 = cv2.imread(r'NTHU_CV_HW1/1a_notredame.jpg')
img_2 = cv2.imread(r'NTHU_CV_HW1/1b_notredame.jpg')
img_3 = cv2.imread(r'NTHU_CV_HW1/chessboard-hw1.jpg')

img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
img_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)


"""
sobel 
"""

def sobel(img):
     # sobel 卷積因子 固定 kernel 
     kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
     kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
     H,W = img.shape
     img_1_gx = signal.convolve2d(img, kernel_x, boundary='symm', mode='same')
     img_1_gy = signal.convolve2d(img, kernel_y, boundary='symm', mode='same')
     
     # magnitude = gx、gy 方向力的大小總和
     #img_1_magnitude = cv2.addWeighted(img_1_gx, 0.5, img_1_gy, 0.5, 0) 
     #img_1_magnitude = img_1_gx + img_1_gy
     
     
     #grad_x=grad_x**2
     #grad_y=grad_y**2
     
     #img_1_theta = np.degrees(np.arctan2(img_1_gx,img_1_gy))
     img_1_magnitude = np.sqrt((img_1_gx ** 2) + (img_1_gy ** 2))
     img_1_direction = np.arctan2(img_1_gy, img_1_gx) * (180 / np.pi) 
     
     #img_1_theta=grad_x+grad_y
     #np.degree()
     
     """
     sobel NMS 
     """
     # img_tmp = np.zeros(img_1.shape)
     # for i in range(1,img_tmp.shape[0]-1):
     #      for j in range(1,img_tmp.shape[1]-1):
     #           if (((img_1_direction[i,j] >= -22.5) and (img_1_direction[i,j]< 22.5)) or
     #           ((img_1_direction[i,j] <= -157.5) and (img_1_direction[i,j] >= -180)) or
     #           ((img_1_direction[i,j] >= 157.5) and (img_1_direction[i,j] < 180)) ):
     #                img_tmp[i,j] = img_1[i,j]
     #           elif ( ( (theta[i,j] >= 22.5) and (theta[i,j]< 67.5) ) or
     #            ( (theta[i,j] <= -112.5) and (theta[i,j] >= -157.5) ) ):
     #                     img_tmp[i,j] = img_1[i,j]
     #           if (img_1_direction[i,j] == 90.0) and  img_1[i,j] == np.max([img_1[i,j],img_1[i,j+1],img_1[i,j-1]]):
     #                     img_tmp[i,j] = img_1[i,j]
     #           if (img_1_direction[i,j] == 45.0) and img_1[i,j] == np.max([img_1[i,j],img_1[i-1,j+1],img_1[i+1,j-1]]):
     #                     img_tmp[i,j] = img_1[i,j]

     return img_1_magnitude ,img_1_direction ,img_1_gx ,img_1_gy #,img_tmp





"""
show
"""
#img_1_magnitude ,img_1_direction , img_1_gx, img_1_gy,img_1_tmp = sobel(img_1)
img_1_magnitude ,img_1_direction , img_1_gx, img_1_gy = sobel(img_3)
# cv2.imwrite("Sobel_img_1_gx.jpg",img_1_gx)
# cv2.imwrite("Sobel_img_1__gy.jpg",img_1_gy)
cv2.imwrite("Sobel_img_3_magnitude.jpg",img_1_magnitude)
cv2.imwrite("Sobel_img_3_direction.jpg",img_1_direction)
#cv2.imwrite("img_1_tmp.jpg",img_1_tmp)

# cv2.waitKey(0)  
# cv2.destroyAllWindows() 


"""
show
"""





(fig, axs) = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
# plot
axs[0].imshow(img_3, cmap="gray")
axs[1].imshow(img_1_magnitude, cmap="hsv")
axs[2].imshow(img_1_direction, cmap="hsv")
# 軸標題
axs[0].set_title("Grayscale")
axs[1].set_title("Gradient magnitude")
axs[2].set_title("Gradient direction")
# 循環遍歷每個軸並關閉 x 和 y 刻度 
for i in range(0, 3):
	axs[i].get_xaxis().set_ticks([])
	axs[i].get_yaxis().set_ticks([])
# show
plt.tight_layout()
plt.show()

