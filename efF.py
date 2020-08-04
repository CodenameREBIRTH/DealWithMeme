# import necessary packages
import numpy as np
import imutils
from imutils import face_utils
from imutils import paths
import dlib
import os
import cv2 as cv
import sys
import shutil


def cover(background, foreground, foregroundmask, coordinates):
    # get foreground dimensions
    (h, w) = foreground.shape[:2]
    (x, y) = coordinates
    # overlay will be of same size as input image
    overlay = np.zeros(background.shape, dtype="uint8")
    overlay[y:y + h, x:x + w] = foreground
    # alpha channel with same size to control transparency
    alpha = np.zeros(background.shape[:2], dtype="uint8")
    alpha[y:y + h, x:x + w] = foregroundmask
    alpha = np.dstack([alpha] * 3)
    # perform alpha blending to merge foreground, background, and alpha channel
    result = alpha_blend(overlay, background, alpha)

    return result


def alpha_blend(foreground, background, alpha):
    # convert foreground, background, alpha layer to float
    foreground = foreground.astype("float")
    background = background.astype("float")
    # scale alpha layer in range [0, 255]
    alpha = alpha.astype("float") / 255

    # alpha blending
    foreground = cv.multiply(alpha, foreground)
    background = cv.multiply(1 - alpha, background)
    result = cv.add(foreground, background)

    return result.astype("uint8")


def gif_maker(inputPath, outputPath, delay, finaldelay, loopamount):
    path = sorted(list(paths.list_images(inputPath)))
    # remove last image path
    lastpath = path[-1]
    path = path[:-1]
    # construct image magick 'convert' command
    command = "convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(path), finaldelay, "".join(lastpath), loopamount, "".join(outputPath))
    #command = "date"
    os.system(command)
    #print(command)

sg = cv.imread(os.path.sep.join(["Assets", "sunglasses.png"]))
sgmask = cv.imread(os.path.sep.join(["Assets", "sunglasses_mask.png"]))

# crete temporary directory if not exist else delete previously created ones
shutil.rmtree("Temporary", ignore_errors=True)
os.makedirs("Temporary")

# load face detector and construct input blob
print("loading model")
detector = cv.dnn.readNetFromCaffe(os.path.sep.join(["FaceDetector", "deploy.prototxt"]), os.path.sep.join(["FaceDetector", "res10_300x300_ssd_iter_140000.caffemodel"]))
predictor = dlib.shape_predictor(os.path.sep.join(["FaceDetector", "shape_predictor_68_face_landmarks.dat"]))
image = cv.imread(os.path.sep.join(["Dataset", "example1.jpg"]))
(h, w) = image.shape[:2]
blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass blob into network
print("computing detection")
detector.setInput(blob)
detection = detector.forward()

# assuming only one face
n = np.argmax(detection[0, 0, :, 2])
confidence = detection[0, 0, n, 2]
if confidence < 0.5:
    print("face not found")
    sys.exit(0)

# compute x, y coordinates of bounding box
box = detection[0, 0, n, 3:7]  * np.array([w, h, w, h])
(x_start, y_start, x_end, y_end) = box.astype("int")

# dlib rectangle object to determine facial landmarks
region = dlib.rectangle(int(x_start), int(y_start), int(x_end), int(y_end))
shape = predictor(image, region)
shape = face_utils.shape_to_np(shape)

(leftstart, leftend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightstart, rightend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
lefteye = shape[leftstart:leftend]
righteye = shape[rightstart:rightend]

# compute center of mass of eye, angle
lefteyecenter = lefteye.mean(axis=0).astype("int")
righteyecenter = righteye.mean(axis=0).astype("int")
angle = np.degrees(np.arctan2(righteyecenter[1]-lefteyecenter[1],
                              righteyecenter[0]-lefteyecenter[0]))-180

# rotate sunglasses with above angle, should just cover the eyes, adjust transparency
sg = imutils.rotate_bound(sg, angle)
sg = imutils.resize(sg, width=int((x_end - x_start) * 0.9))
sgmask = cv.cvtColor(sgmask, cv.COLOR_BGR2GRAY)
sgmask = cv.threshold(sgmask, 0, 255, cv.THRESH_BINARY)[1]
sgmask = imutils.rotate_bound(sgmask, angle)
sgmask = imutils.resize(sgmask, width=int((x_end - x_start) * 0.9), inter=cv.INTER_NEAREST)

# determine equal spaces from top of frame to desired location
step = np.linspace(0, righteyecenter[1], 20, dtype="int")

for (x, y) in enumerate(step):
    x_shift = int(sg.shape[1] * 0.25)
    y_shift = int(sg.shape[0] * 0.35)
    y = max(0, y - y_shift)

    answer = cover(image, sg, sgmask, (righteyecenter[0] - x_shift, y))
    if x == len(step) - 1:
        meme = cv.imread(os.path.sep.join(["Assets", "meme.png"]))
        mememask = cv.imread(os.path.sep.join(["Assets", "meme_mask.png"]))
        mememask = cv.cvtColor(mememask, cv.COLOR_BGR2GRAY)
        mememask = cv.threshold(mememask, 0, 255, cv.THRESH_BINARY)[1]
        meme = imutils.resize(meme, width=int(w * 0.8))
        mememask = imutils.resize(mememask, width=int(w * 0.8), inter=cv.INTER_NEAREST)
        answer = cover(answer, meme, mememask, (int(w * 0.1), int(h * 0.8)))

    # write output image to temporary directory
    path = os.path.sep.join(["Temporary", "{}.jpg".format(str(x).zfill(8))])
    cv.imwrite(path, answer)

# crete output GIF image and clean temporary directory
print("creating GIF image")
gif_maker("Temporary", "OutputImage.gif", 5, 250, 0)
#print("cleaning directory")
#shutil.rmtree("Temporary", ignore_errors=True)