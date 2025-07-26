# webcam
from itertools import count
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
cap=cv2.VideoCapture(0) # Start the webcam
detector=HandDetector(maxHands=1) # Detect only one hands
offset=20 # Padding around the detected hand
imgSize=300 # The final image size (300x300)
folder = "images/z"  # Folder where processed images will be saved
count=0; # Counter for saved images

while True:
    success, img =cap.read() # Read a frame from the webcam
    Hands ,img = detector.findHands(img) # Detect hands
    if Hands:
        hand=Hands[0] #check for one hand only
        x,y,w,h=hand['bbox'] # Get bounding box coordinates (x, y, width, height)
        imgUse=np.ones((imgSize,imgSize,3),np.uint8)*255 # White background image
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset] # Crop hand region
        aspectRatio=h/w   # Calculate aspect ratio (height/width)
        if aspectRatio > 1:
            k=imgSize/h  # Scaling factor based on height
            WCal=math.ceil(k*w)  # Calculate new width
            imgResize=cv2.resize(imgCrop,(WCal,imgSize))  # Resize image
            imgResizeShape = imgResize.shape
            wGap=math.ceil((imgSize-WCal)/2)   # Center horizontally
            imgUse[0:imgResizeShape[0], wGap:WCal+wGap] = imgResize  # Place in the center
        else :
            k=imgSize/w
            HCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,HCal))
            imgResizeShape = imgResize.shape
            HGap=math.ceil((imgSize-HCal)/2)  # Center vertically
            imgUse[HGap:HCal+HGap,:] = imgResize
        cv2.imshow("Imageused",imgUse)  # Show the processed image

    cv2.imshow("Image",img)  # Show the original webcam feed
    key=cv2.waitKey(1)
    if key==ord("s"):  #saving the image when s is pressed
        count+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgUse)# saving images
        print(count)
