import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy import misc
import math
import sys 

Sigma_1=5

#讀取照片
img_1 = cv2.imread(r'NTHU_CV_HW1/1a_notredame.jpg')
img_2 = cv2.imread(r'NTHU_CV_HW1/1b_notredame.jpg')
img_3 = cv2.imread(r'NTHU_CV_HW1/chessboard-hw1.jpg')
cv2.imshow('img_1',img_1)
cv2.imshow('img_2',img_2)
cv2.imshow('img_3',img_3)

# check
if img_1 is None:
    print("Could not open or find the image")
    sys.exit()
if img_2 is None:
    print("Could not open or find the image")
    sys.exit()
if img_3 is None:
    print("Could not open or find the image")
    sys.exit()



# 灰階

gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
gray_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray_1',gray_1)

"""
sigma = 5 and kernel size=5 and 10 Gassian filter 
建 kernel 大小
kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
"""
x_1, y_1 = np.mgrid[0:5, 0:5]
x_2, y_2 = np.mgrid[0:10, 0:10]


""" 
Gaussian Filter 數學函式
ref. https://medium.com/@bob800530/python-gaussian-filter-%E6%A6%82%E5%BF%B5%E8%88%87%E5%AF%A6%E4%BD%9C-676aac52ea17
"""
gaussian_kernel_1 = (1/(2*Sigma_1**2*math.pi))*(np.exp(-(x_1**2+y_1**2)/(2*Sigma_1**2)))
gaussian_kernel_2 = (1/(2*Sigma_1**2*math.pi))*(np.exp(-(x_2**2+y_2**2)/(2*Sigma_1**2)))
#gaussian_kernel_1 = (np.exp(-(x_1**2+y_1**2)))
#gaussian_kernel_2 = (np.exp(-(x_2**2+y_2**2)))

"""
ref. https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

np.square : array 的每個元素都取平方並回傳

"""
def gaussian_kernel(sigma, size):
    mu = np.floor([size / 2, size / 2])
    size = int(size)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(0.5/(sigma*sigma)) * (np.square(i-mu[0]) + 
            np.square(j-mu[0]))) / np.sqrt(2*math.pi*sigma*sigma)
    kernel = kernel/np.sum(kernel)
    return kernel


# Normalization
gaussian_kernel_1 = gaussian_kernel_1 / gaussian_kernel_1.sum()
gaussian_kernel_2 = gaussian_kernel_2 / gaussian_kernel_2.sum()


"""
grad  卷積
"""
# test ref. code
cov_test = signal.convolve2d(gray_1, gaussian_kernel(5,5), boundary='symm', mode='same')

# size = 5 metrix
cov_1 = signal.convolve2d(gray_1, gaussian_kernel_1, boundary='symm', mode='same')
cov_2 = signal.convolve2d(gray_2, gaussian_kernel_1, boundary='symm', mode='same')
cov_3 = signal.convolve2d(gray_3, gaussian_kernel_1, boundary='symm', mode='same')

# size = 10 metrix
cov_1_1 = signal.convolve2d(gray_1, gaussian_kernel_2, boundary='symm', mode='same')
cov_2_1 = signal.convolve2d(gray_2, gaussian_kernel_2, boundary='symm', mode='same')
cov_3_1 = signal.convolve2d(gray_3, gaussian_kernel_2, boundary='symm', mode='same')



"""
show img
"""
cv2.imshow("cov_test",cov_test)
#cv2.imshow("cov_1",cov_1)
#plt.imshow(cov_1, interpolation='nearest')
# cv2.imshow('cov_1.jpg',cov_1)
cv2.imwrite('GS_size5_test_img.jpg',cov_test)
cv2.imwrite('gray_img1.jpg',gray_1)

# size = 5 metrix
cv2.imwrite('GS_size5_img1.jpg',cov_1)
cv2.imwrite('GS_size5_img2.jpg',cov_2)
cv2.imwrite('GS_size5_img3.jpg',cov_3)

# size = 10 metrix
cv2.imwrite('GS_size10_img1.jpg',cov_1_1)
cv2.imwrite('GS_size10_img2.jpg',cov_2_1)
cv2.imwrite('GS_size10_img3.jpg',cov_3_1)

cv2.waitKey(0)  
cv2.destroyAllWindows() 