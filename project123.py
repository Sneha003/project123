import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=2500,train_size=7500,random_state=9)

X_train_Scaled=X_train/255
X_test_Scaled=X_test/255

classifier= LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_Scaled,y_train)

y_pred = classifier.predict(X_test_Scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is :- ",accuracy)


cap= cv2.VideoCapture(0)
while(True):
    try:
        rect,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #qprint(gray.shape)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))

        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),3)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        #print(roi)
        image_pil=Image.fromarray(roi)

        im_btw=image_pil.convert("L")

        im_btw_resized= im_btw.resize((22,30),Image.ANTIALIAS)
        
        #print(im_btw_resized)
        
        im_btw_resized_inverted=PIL.ImageOps.invert(im_btw_resized)
        #print(im_btw_resized_inverted)
        pixel_filter=20
        mixpixel_filter=np.percentile(im_btw_resized,pixel_filter)

        image_bw_resized_inverted_scaled = np.clip(im_btw_resized-mixpixel_filter, 0, 255)
        #print(image_pil)
        max_pixel = np.max(im_btw_resized)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660) 
        test_pred = classifier.predict(test_sample)
        print("predicted digit is ",test_pred)
        cv2.imshow("Digit",gray)
        
        if(cv2.waitKey(1)==ord("q")):
            break

    except Exception as e: 
        pass
        
cap.release()
cv2.destroyAllWindows()


