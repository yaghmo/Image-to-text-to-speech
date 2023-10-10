import cv2
import numpy as np
import cvzone
from paragraph import seg_paragraphe
from text import seg_text
from char import char_seg
from prediction import predict_word
from NLP_scrap import NLP
from tqdm import tqdm
import os
from tensorflow.keras.models import load_model

'''
NOTE : all the modules can be launched separatly.
For every called function of a module, its input has already got stored in the folder output
for a better visibility and understanding of the process we suggest you to simply set the if to False when you reach block comment section named `BLOCK` starting at line 200 at the end of the this code 
and run the modules independently in this order ( main.py -> paragraph.py -> text.py -> char.py -> prediction.py -> NLP_scrap.py )

'''


def biggest_contour(contours):
    # finding the biggest contour, which has to be closed and if its the case it returns the 4 corners (rectangle) #
    biggest = np.array([])
    max_area = 0
    for i in contours:
        # calculate the area of each contour
        area = cv2.contourArea(i)
        # only check those who are big enough that may represent a board #
        if area > 150000:
            # get the perimetre #
            peri = cv2.arcLength(i, True)
            # getting a bounding corners #
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


'''
some good parameters for the function seg_parag()
image6. kern=[7,2], it=12
image 7. rip kern=[5,1],it=12
image 8. kern=[3,4],it=18
image 2. kern=[4,2],it=20
image 9. kern=[7,2],it=20
image 11. kern=[5,2],it=20
image 12. kern=[3,3],it=12
'''

folder_name = "output"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# resize the image #
img = cv2.resize(cv2.imread('images/3.jpg'), (1080, 720))
img_original = img.copy()


# Image preprocess #
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 20, 30, 30)
edged = cv2.Canny(gray, 20, 20)


cv2.imshow('Gray image with a bilateral filter', gray)
cv2.imwrite('output/gray.png', gray)
cv2.imshow('Edges detected with canny', edged)
cv2.imwrite('output/edged.png', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Contour detection #
contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# getting the 10 1st biggest contours #
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
cv2.imshow('The few 1st biggest contours', img_contours)
cv2.imwrite('output/img_contours.png', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


# getting the biggest one #
biggest = biggest_contour(contours)


# warp image size #
max_width = 1080
max_height = 720

# board corners points #
if len(biggest) == 4:
    # if the biggest shape has 4 corners #
    points = biggest.reshape(4, 2)
else: 
    # if no rectangle shape like found #
    points = np.zeros((4, 2), dtype="float32")

# new points as ref for the wrapping #
input_points = np.zeros((4, 2), dtype="float32")


img_select = img_original.copy()
pointed = img_original.copy()
X, Y = 0, 0
counter = 0


def mouse_callback(event, x, y, flags, param):
    '''
    NOTE: redifinition of mouse_callback, it gets triggered after left button is pressed down
    used to click on the window in order to choose your corners by pressing the key `s` to save a selecter corner
    '''
    global img_select, pointed, X, Y, counter

    if event == cv2.EVENT_LBUTTONDOWN:
        # if chosen detected objects #
        img_select = pointed.copy()
        cvzone.putTextRect(
            img_select, f'C :{counter +1}', (x, max(0, y-5)), thickness=1, scale=1, offset=3)
        X, Y = x, y
        cv2.imshow('Corner selection', img_select)


if len(biggest) != 4:
    # only occures when theres no detected board => manual region selection #
    print('We couldnt detect any complete board displated on the image, please choose the 4 corners to select a regrion, by pressing the key s to submit the corner')
    cv2.namedWindow("Corner selection")
    cv2.imshow('Corner selection', img_original)
    cv2.setMouseCallback("Corner selection", mouse_callback)
    while True:
        key = cv2.waitKey(0)
        if key == ord('s'):
            points[counter] = [X, Y]
            counter += 1
            pointed = img_select.copy()
            cv2.imwrite(f'output/{counter+1}.png', pointed)
        if counter == 4:
            cv2.destroyAllWindows()
            break

# reordering the corners' coordinates #
points_sum = points.sum(axis=1)
input_points[0] = points[np.argmin(points_sum)]
input_points[3] = points[np.argmax(points_sum)]

points_diff = np.diff(points, axis=1)
input_points[1] = points[np.argmin(points_diff)]
input_points[2] = points[np.argmax(points_diff)]


if len(biggest) == 4:
    # shows the found corners when a detectable board is on the image #
    img_biggest = img.copy() 
    gray_color = gray.copy()
    gray_color = cv2.cvtColor(gray_color, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_biggest, [biggest], -1, (0, 255, 0), 3)
    cv2.drawContours(gray_color, biggest, -1, (0, 0, 255), 2)
    cv2.imshow('Biggest contour', img_biggest)
    cv2.imshow('Biggest contour on gray scale', gray_color)
    cv2.imwrite('output/img_biggest.png', img_biggest)
    cv2.imwrite('output/gray_color.png', gray_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Perspective transformation #
matrix = cv2.getPerspectiveTransform(input_points, np.float32(
    [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]))
warp = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

cv2.imshow("Warped perspective", warp)
cv2.imwrite('output/warp.png', warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

key = input(
    "Do you want to paragraphe segment it ? (press 'y' to confirm / 'n' to decline)")
while key not in ['y','n']:
    key = input("Press a valide key please \n")


warp_detection = True
if key == 'n':
    warp_detection = False



'''
#**********************#
#********BLOCK*********#
#**********************#
'''
# set this to False to run modules independently #
if True:
    # if we decide to parag seg then we call the function else the image stays it self #
    paragraphs = seg_paragraphe(warp, kern=[7,2], it=20) if warp_detection else [warp]


    # get words by lines in every paragraph #
    for paragraph in paragraphs:
        lines = seg_text(paragraph)

    # if words exist #
    if lines != None:
        text = ''
        label_list = np.array(['0', '1', '2', '3', '4', '5', '6',
                        '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
                        'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                        'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 
                        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
                        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
                        't', 'u', 'v', 'w', 'x', 'y', 'z'])

        # loading the stinky model #
        model = load_model('weights\my_cnn_model.h5')
        for l,words in enumerate(lines):
            text_line = []
            for w, word in enumerate(words):
                chars = char_seg(word)
                prediction = predict_word(chars,model=model)
                text_word = ''
                for i in tqdm(prediction):
                    text_word += label_list[i]
                text_line.append(text_word)
                print(f'Word No. {w+1} of line No. {l+1} generated: {text_word}')
            print(f"Line No. {l+1} generated: {' '.join(text_line)}")
            text = ' '.join([text, ' '.join(text_line)])

    NLP(text)
