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

"""
非極大值
去除假的邊緣響應
"""
# def NMS(theta,img_1):
img_1_NMS = np.zeros(img_1.shape)
for i in range(1,img_tmp.shape[0]-1):
     for j in range(1,img_tmp.shape[1]-1):
          if (img_1_theta[i,j] == 0.0) and (img_1[i,j] == np.max([img_1[i,j],img_1[i+1,j],img_1[i-1,j]]) ):
                    img_tmp[i,j] = img_1[i,j]
          if (img_1_theta[i,j] == -45.0) and img_1[i,j] == np.max([img_1[i,j],img_1[i-1,j-1],img_1[i+1,j+1]]):
                    img_tmp[i,j] = img_1[i,j]
          if (img_1_theta[i,j] == 90.0) and  img_1[i,j] == np.max([img_1[i,j],img_1[i,j+1],img_1[i,j-1]]):
                    img_tmp[i,j] = img_1[i,j]
          if (img_1_theta[i,j] == 45.0) and img_1[i,j] == np.max([img_1[i,j],img_1[i-1,j+1],img_1[i+1,j-1]]):
                    img_tmp[i,j] = img_1[i,j]
 
"""
雙門檻與連接
"""
# max_threshold=100
# min_threshold=10
# max_value=255
# min_value=0
# h = img_1.shape[0]
# w = img_1.shape[1]
# # for i in range(h):
# #     for j in range(w):
# #         if img[i,j]>=max_threshold:
# #             img[i,j]= max_value
# #         else:
# #             img[i,j]= min_v


"""
確認 9 格位置 是否大於 max_threshold
"""
# find_x=[-1,-1,-1,1,1,1,0,0,0]
# find_y=[-1,0,1,-1,0,1,-1,0,1]
# for i in range (h*w):
#     if img_1[i+find_x[0],j+find_y[0]]>max_threshold:
#         img_1[i,j]=max_value
#     if img_1[i+find_x[1],j+find_y[1]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[2],j+find_y[2]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[3],j+find_y[3]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[4],j+find_y[4]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[5],j+find_y[5]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[6],j+find_y[6]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[7],j+find_y[7]]>max_threshold:
#          img_1[i,j]=max_value
#     if img_1[i+find_x[8],j+find_y[8]] >max_threshold:
#          img_1[i,j]=max_value

cv2.imwrite("Sobel_img_1__gx.jpg",img_1_gx)
cv2.imwrite("Sobel_img_1__gy.jpg",img_1_gy)
cv2.imwrite("Sobel_img_1_magnitude.jpg",img_1_magnitude)
cv2.imwrite("Sobel_img_1_theta.jpg",img_1_theta)

cv2.waitKey(0)  
cv2.destroyAllWindows() 