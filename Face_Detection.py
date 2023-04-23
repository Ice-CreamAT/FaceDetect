import cv2

image = cv2.imread('Jobs.png')                                             # Select the image file that the code will read

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Select the haarscascade parameter.
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                       # Convect the image to grey scale.

# Detection face in Bounding Boxes.
detections = classifier.detectMultiScale(grey_image, scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(30,30),
                                           maxSize=(1000,1000))

# ScaleFactor: resize a larger object to a smaller one. Slower if the value is smaller.
# minNeighbors: How many neighbors must each candidate rectangle have to keep it. Higher values = higher quality.
# minSize: Specifies the smallest object to be recognized.
# maxSize: (I think it's already obvious.)

print(detections)             # Show the positions of each face in the image
print(len(detections))        # Show the number of detected faces

 # Each face has an "x" and "y" coordinate.
 # "w" and "h" are for width and height
for (x, y, w, h) in detections:
     # Parameters for drawing the detection rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

cv2.imshow('Face Detection', image)      # Show image with title
cv2.waitKey(0)                           # Will close the window when you press any key
# Free up windows from memory
cv2.destroyAllWindows()
