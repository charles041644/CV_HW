import numpy as np

import cv2

img_1 = cv2.imread(r'NTHU_CV_HW1/1a_notredame.jpg')
img_2 = cv2.imread(r'NTHU_CV_HW1/1b_notredame.jpg')

img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # 計算特徵

kp1,des1 = sift.detectAndCompute(img_1, None)
kp2,des2 = sift.detectAndCompute(img_2 ,None)

# kp = sift.detect(gray_1,None)
# kp = sift.detect(gray_1,None)

# 4) Flann特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    # if m.distance < 0.50*n.distance:
    if m.distance <  n.distance:
        goodMatch.append(m)
goodMatch = np.expand_dims(goodMatch, 1)    
img_out_100_knn = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goodMatch[:100], None, flags=2)

# draw the detect interest points 
# img_1 = cv2.drawKeypoints(gray_1,goodMatch[:100],img_1)

#cv2.imshow('img',img_1)
#cv2.waitKey()
#cv2.imwrite("img_out_100_knn.jpg",img_out_100_knn)

cv2.waitKey(0)
cv2.destroyAllWindows()