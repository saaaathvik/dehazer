import numpy as np
import cv2
from tensorflow.keras.models import load_model

# initialize the HOG descriptor/person detector
def preprocess(img):
  input_shape = (480, 640, 3)
  img = cv2.resize(img, (input_shape[1], input_shape[0]))
  img = img/255.0
  img = np.expand_dims(img, axis=0)
  return img
model = load_model("dehazer1.h5")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    img = preprocess(frame)
    output_image = model.predict(img)
    # cv2_imshow(output_image)
    # output_image = cv2.cvtColor((output_image[0] * 255).astype(np.uint8),cv2.COLOR_BGR2RGB)
    output_image = (output_image[0] * 255).astype(np.uint8)
    #using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(output_image, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(output_image, (xA, yA), (xB, yB),
                      (0, 255, 0), 2)

    # Write the output video
    out.write(frame.astype('uint8'))
    combined_frame = np.hstack((frame, output_image))
    # Display the resulting frame
    cv2.imshow('frame', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)