import cv2

path = "dataset-images/000bd8f2-0a69-421b-925e-c8aa695907b6.jpg"

def showImg(img):
    cv2.imshow("dice", img)
    cv2.waitKey(0)

def getContours(img):
    pipCount = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        par = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*par, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pipCount += getPips(original[y:y+h, x:x+w], y, x+w)     
    return pipCount
  
def getPips(img, y, x):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7,7), 5)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    #cv2.imshow("pips", thresh)
    cv2.waitKey(0)
    keypoints = detector.detect(thresh)
    numberOfKeypoitns = len(keypoints)
    cv2.putText(final_image, str(numberOfKeypoitns), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    return numberOfKeypoitns



params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 150
params.maxThreshold = 200

params.filterByCircularity = True
params.minCircularity = 0.1


params.filterByInertia = True
params.minInertiaRatio = 0.5

params.filterByConvexity = True
params.minConvexity = 0

detector = cv2.SimpleBlobDetector_create(params)


        
           
original = cv2.imread(path)        
showImg(original)
final_image = original.copy()
img_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#showImg(img_gray)
img_blur = cv2.GaussianBlur(img_gray, (7,7), 5)
#showImg(img_blur)
_, tresh = cv2.threshold(img_blur, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


pips = getContours(tresh)
cv2.putText(final_image, str(pips), (0,final_image.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0))

showImg(tresh)
showImg(final_image)


cv2.destroyAllWindows()





    
