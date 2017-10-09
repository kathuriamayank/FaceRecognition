#Import the required modules
import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
path="/home/mayank/Documents/all/MachineLearning/OpenCV/FaceRecognition/yalefaces"
path2="/home/mayank/Desktop/images"



#Load the face detection Cascade
facecascade=cv2.CascadeClassifier("/home/mayank/Documents/all/MachineLearning/OpenCV/FaceDetection/haarcascade_frontalface_default.xml")


#Create the Face Recognizer Object
recognizer = cv2.createLBPHFaceRecognizer()


#Creating The Function Now

def get_images_labels(path):
    images=[]
    labels=[]
    images_paths=[]
    
    for ix in os.listdir(path):
        if not ix.endswith(".sad"):
            images_paths.append(ix)
    
    
    #Storing Labels Now
    for ix in images_paths:
        labels.append(int(ix.split(".")[0].replace("subject","")))
        #Now Add Faces To Image By Making Use of Haar Cascade
        image_pil = Image.open(os.path.join(path,ix)).convert('L')
        img = np.array(image_pil, 'uint8')
        #img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        
        face=facecascade.detectMultiScale(img,1.3,5)
        for (x,y,w,h) in face:
            roi=np.array(img[y:y+h,x:x+w])
            images.append(roi)
            cv2.imshow("Image",roi)
            cv2.waitKey(30)
    labels=np.array(labels)
    images=np.array(images)
    
    return images,labels


images,labels=get_images_labels(path)
cv2.destroyAllWindows()
print type(images)
print labels

