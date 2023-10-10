import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import find_peaks, peak_widths



def seg_paragraphe(img, max_width = 1080, max_height = 720, edge=[100,60], kern=[10,0], it=20):
    '''
    Functionality:
    This function is used to segment a paragraph of text into words which will be stored in the output folder
    NOTE: Some of the parameters might need to be changed depending on the image

    Input : 
    img | type = uint8
    max_width | type = int
    max_height | type = int
    edge | type = array of size 2: int, will be the parameters of the canny methode
    kern | type = array of size 2: int, represent the kernel size for the dilatation methode ( vertical focus )
    it | type = int , Nb. of iterations for the same methode ( dilatation )

    Output:
    either the list of the paragraphes that had been found of the image it self if nothing was found
    '''
    img = cv2.resize(img, (max_width, max_height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img, (0, 0), 3)
    
    # ↑↑removing 80% of the smoothed image called gray ↑↑
    mask = cv2.addWeighted(img, 1, gray, -0.8, 2)

    edged = cv2.Canny(mask, edge[0], edge[1])
    edged[max_height-20:,:] = 0
    edged[:,:20] = 0
    edged[:,max_width-20:] = 0
    edged[:20,:] = 0
    kernel = np.ones((kern), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=it)
    cv2.destroyAllWindows()

    img_contours = img.copy()
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour_parag = sorted(contours, key=lambda cntr: cv2.boundingRect(cntr)[0])
    parag_list = []
    for parag in sorted_contour_parag:
        x,y,w,h = cv2.boundingRect(parag)
        
        area = cv2.contourArea(parag, True)
        
        if w*h >3000  and w*h < max_height*max_height*0.7 and np.sum(dilated[y:y+h, x:x+w])/255 < h*w*0.9 and h < max_height*0.8 and w < max_width*0.8:
            cv2.rectangle(img_contours, cv2.boundingRect(parag), (0, 0, 255), 1)
            parag_list.append(img[y:y+h, x:x+w])
   
    cv2.imshow('Paragraphes detected on the image',img_contours)
    cv2.imwrite('output/Paragraphes detected on the image.png',img_contours)
    for i in range(len(parag_list)):
        cv2.imshow(f'The Paragraphe No. {i+1}',parag_list[i])
        cv2.imwrite(f'output/The Paragraphe No. {i+1}.png',parag_list[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return parag_list if len(parag_list) > 0 else img



if __name__ == "__main__":

    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        
    img = cv2.imread('output\warp.png')
    cv2.imshow('sdf',img)
    cv2.waitKey(0)
    seg_paragraphe(img)
