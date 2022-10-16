import cv2
import numpy as np
import sys 
#from psd_tools import PSDImage

# # 1) psd to png
# psd1 = PSDImage.load('200x800.ai.psd')
# psd1.as_PIL().save('psd_image_to_detect1.png')

# psd2 = PSDImage.load('800x200.ai.psd')
# psd2.as_PIL().save('psd_image_to_detect2.png')

# 2) 以灰度图的形式读入图片

img_1 = cv2.imread(r'NTHU_CV_HW1/1a_notredame.jpg')
img_2 = cv2.imread(r'NTHU_CV_HW1/1b_notredame.jpg')

if img_1 is None:
    print("Could not open or find the image")
    sys.exit()
if img_2 is None:
    print("Could not open or find the image")
    sys.exit()   

img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)


# 3) SIFT特征计算
sift = cv2.xfeatures2d.SIFT_create()
# descriptors : Computed descriptors. Output concatenated vectors of descriptors. Each descriptor is a 128-element vector, as returned by cv.SIFT.descriptorSize,
# so the total size of descriptors will be numel(keypoints) * obj.descriptorSize(). A matrix of size N-by-128 of class single, one row per keypoint. 
kp1, des1 = sift.detectAndCompute(img_1, None)
kp2, des2 = sift.detectAndCompute(img_2, None)

# 4) Flann特征匹配
"""
FLANN 匹配器有两个参数，一个是indexParams，另一个是searchParams，以字典的形式进行参数传递。为了计算匹配，FLANN内部会决定如何处理索引和搜索对象。
checks，表示制定索引树要被遍历的次数
5 kd—trees，50 checks 可以取得较好的匹配精度，并且可以在较短的时间内完成。
"""
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
	# goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
    if m.distance < 0.50*n.distance:
        goodMatch.append(m)
# 增加一个维度
goodMatch = np.expand_dims(goodMatch, 1)
# print(goodMatch[:20])


#img_out = cv2.drawMatchesKnn(psd_img_1, psd_kp1, psd_img_2, psd_kp2, goodMatch[:15], None, flags=2)
img_out_100_knn = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goodMatch[:100], None, flags=2)
img_out_10_knn = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goodMatch[:10], None, flags=2)

# cv2.imshow('image', img_out)
cv2.imwrite("SIFT_100_knn.jpg",img_out_100_knn)
cv2.imwrite("SIFT_10_knn.jpg",img_out_10_knn)
cv2.waitKey(0)
cv2.destroyAllWindows()
