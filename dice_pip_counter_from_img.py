import cv2

path = "pictures/img2.jpg"

def showImg(img):
    img = cv2. resize(img, (600, 500))
    cv2.imshow("dice", img)
    cv2.waitKey(0)

def getContours(thresh_img, result_image, blobDetector):
    pipCount = 0
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        par = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*par, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        if((w)*(h) > thresh_img.shape[0]*thresh_img.shape[1]*0.01):
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            pipCount += getPips(thresh_img[y:y+h, x:x+w], y, x+w, blobDetector, result_image)     
    return pipCount
  
def getPips(img, y, x, blobDetector, result_image):
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_blur = cv2.GaussianBlur(img_gray, (7,7), 5)
    # _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    cv2.imshow("pips", img)
    cv2.waitKey(0)
    keypoints = blobDetector.detect(img)
    numberOfKeypoitns = len(keypoints)
    cv2.putText(result_image, str(numberOfKeypoitns), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255))
    return numberOfKeypoitns

def getBlobDetector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 150
    params.maxThreshold = 200
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0
    return cv2.SimpleBlobDetector_create(params)

           
original = cv2.imread(path)        
showImg(original)
result_image = original.copy()
img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
showImg(img_gray)
img_blur = cv2.GaussianBlur(img_gray, (7,7), 5)
showImg(img_blur)
_, tresh = cv2.threshold(img_blur, 210, 245, cv2.THRESH_BINARY)


blobDetector = getBlobDetector()
pips = getContours(tresh, result_image, blobDetector)

cv2.putText(result_image, str(pips), (0,result_image.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0))

showImg(tresh)
showImg(result_image)
print(pips)


cv2.destroyAllWindows()





    
