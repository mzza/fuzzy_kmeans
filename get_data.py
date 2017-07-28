import cv2

def get_image_data(path):
    image_data=cv2.imread(path,0)
    return image_data

if __name__=="__main__":
    image_matrix=get_image_data('/home/lza/Documents/FUZZY_KMEANS/gourd.bmp')
    print(image_matrix)
    cv2.namedWindow("image")
    cv2.imshow("image",image_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
