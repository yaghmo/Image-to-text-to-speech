import cv2
import numpy as np
from skimage.transform import resize
import os
from tensorflow.keras.models import load_model

def pre_proc(img, max_width=256, max_height=256):

    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    img = resize(img, (max_width, max_height), mode='constant',
                 preserve_range=True).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('Image',img)
    cv2.waitKey(0)
    return img


def prediction(to_predict, model):
    return np.argmax(model.predict([to_predict]))


def predict_word(img, model=None):
    word = []
    for char in img:
        to_predict = pre_proc(char, 256, 256)
        word.append(prediction(to_predict.reshape(1,256,256,1), model))
    return word


if __name__ == "__main__":
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    label_list = np.array(['0', '1', '2', '3', '4', '5', '6',
                           '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                           'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                           'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                           'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                           'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                           't', 'u', 'v', 'w', 'x', 'y', 'z'])
    image = [cv2.imread('output\Charater No. 1.png', 0)]


    model = load_model('weights\my_cnn_model.h5')
    pred = predict_word(image,model)
    word = ''
    for i in pred:
        word += label_list[i]
    print(word)
