#from pudb import set_trace;set_trace()
import get_data
import cv2
import fuzzy_kmeans

image_data=get_data.get_image_data('/home/lza/Documents/FUZZY_KMEANS/gourd.bmp')

clf=fuzzy_kmeans.Fuzzy_Kmeans('array')
clf.fit(image_data)
label=clf.predict()

seg_image=image_data
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        if label[i,j]==1:
            seg_image[i,j]=10
        else:
            seg_image[i,j]=200

cv2.imwrite("seg_image.bmp",seg_image)

cv2.namedWindow("image_seg")
cv2.imshow("image_seg",seg_image)
cv2.waitKey(0)

