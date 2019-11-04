import sys
import cv2


# read image
image = cv2.imread(filename = "./uploads/" + sys.argv[1])

# display image and wait for keypress, using a resizable window
cv2.namedWindow(winname = "image", flags = cv2.WINDOW_NORMAL)
cv2.imshow(winname = "image", mat = image)
cv2.waitKey(delay = 0)

print("Filename for Uploaded Image: " + sys.argv[1])
