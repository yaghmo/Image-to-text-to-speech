'''

NOTE: function like peaks finder of scipy
#gradient movement towards the peaks and count them
def histogramme(treshold,hist,minval,size,context,p):    
    nbc=[]
    pref=[]
    suf=[]
    histresh=[]

    hist=hist.astype(int)

    for i in range(treshold):
        pref=np.append(pref,hist[0])
        suf=np.append(suf,hist[size-1])

    pref=pref.astype(int)
    suf=suf.astype(int)

    pref=np.append(pref,hist)
    histresh = np.append(pref,suf)
    for i in range(treshold, size+treshold):
        if(histresh[i] == np.max(histresh[i-treshold:i+treshold]) and histresh[i]>minval):
            print(f'A {context} has been detected around the pixel {p} = :',i-treshold)
            nbc=np.append(nbc,i-treshold)
            i+=treshold
        elif(np.argmax(histresh[i-treshold:i+treshold]>i)):
            i=np.argmax(histresh[i-treshold:i+treshold])-1
    return nbc.astype(int)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import find_peaks, peak_widths


def seg_text(img, max_width = 1080, max_height=720, edge=[100,60], distL = 100, distW =20):
    '''
    Functionality:
    This function is used to segment a paragraphe of text into lines then words which will be stored in the output folder
    NOTE: Some of the parameters might need to be changed depending on the image

    Input : 
    img | type = uint8
    max_width | type = int
    max_height | type = int
    Distl | type = array of size 2: int, will be the parameters of the canny methode
    distL, distW | type = int , maximum range of finding the peak on axis x of the histogram 
    ( this doesnt mean the horizontal hist which calculates the occorurences on the axis X of the image )

    Output:
    either the list of the paragraphes that had been found of the image it self if nothing was found
    '''

    img = cv2.resize(img, (max_width, max_height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img, (0, 0), 3)
    mask = cv2.addWeighted(img, 1, gray, -0.8, 2)
    

    edged = cv2.Canny(mask, edge[0], edge[1])
    edged_words = edged.copy()

    kernel = np.ones((1, 10), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=3)

    kernel = np.ones((5, 1), np.uint8)
    edged_words = cv2.dilate(edged_words, kernel, iterations=3)

    cv2.imshow('Delated image',dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    hist_x = np.sum(dilated,1)/255
    hist_y = np.sum(dilated,0)/255
    
    lines, _ = find_peaks(hist_x.flatten(), distance=distL, height=150)
    words, _ = find_peaks(hist_y, distance=distW, height=100)

    fig=plt.figure("X and Y histogramms",figsize=(10, 8))
    fig.add_subplot(2,1, 1)
    plt.title('X Histogramme')
    plt.plot(lines, hist_x[lines], 'ro')
    plt.plot(hist_x,label='X')
    plt.xlabel('Pixel')
    plt.ylabel('Occurence')
    fig.add_subplot(2,1, 2)
    plt.title('Y Histogramme')
    plt.plot(words, hist_y[words], 'rx')
    plt.plot(hist_y,label='Y')
    plt.xlabel('Pixel')
    plt.ylabel('Occurence')
    plt.show()


    print('Estimated lignes on the image is :', len(lines))

    print('Those lines are around these pixels :', lines)


    print('Estimated words on the image is :', len(words))

    if len(lines) > 0:
        crops_ori, crops_edged, crops_word = [], [], []
        crops_ori.append(img[0:(lines[0]+lines[1])//2, :])
        crops_edged.append(edged[0:(lines[0]+lines[1])//2, :])
        crops_word.append(img[0:(lines[0]+lines[1])//2, :].copy())
        # segmentation

        for i in range(1, len(lines)-1):
            crops_ori.append(img[(lines[i]+lines[i-1]) //2:(lines[i]+lines[i+1])//2, :])
            crops_edged.append(edged[(lines[i]+lines[i-1])//2:(lines[i]+lines[i+1])//2, :])
            crops_word.append(img[(lines[i]+lines[i-1]) //2:(lines[i]+lines[i+1])//2, :].copy())

        crops_ori.append(img[(lines[len(lines)-2]+lines[len(lines)-1])//2:max_width, :])
        crops_edged.append(edged[(lines[len(lines)-2]+lines[len(lines)-1])//2:max_width, :])
        crops_word.append(img[(lines[len(lines)-2]+lines[len(lines)-1])//2:max_width, :].copy())

        
        '''

        NOTE: the following kernel size depend on the word, so its adjustable as needed
        
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 7))
        for i in range(len(crops_edged)):
            crops_edged[i] = cv2.dilate(crops_edged[i], kernel, iterations=2)

        

        lines_list = []
        word_count = 0

        for i in range(len(crops_edged)):
            
            contours, hierarchy = cv2.findContours(
                crops_edged[i].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x, y, w, h = cv2.boundingRect(crops_edged[i])
            sorted_contour_words = sorted(
                contours, key=lambda cntr: cv2.boundingRect(cntr)[0])
            words_list = []
            word_crop = []
            for word in sorted_contour_words:
                '''
                NOTE: Variable paramete
                '''
                if cv2.contourArea(word) > 1000:
                    words_list.append(cv2.boundingRect(word))
                    x, y, w, h = cv2.boundingRect(word)
                    word_crop.append(crops_word[i][y:y+h, x:x+w])
                    cv2.rectangle(crops_ori[i], cv2.boundingRect(word), (0, 0, 255), 1)
            word_count += len(words_list)
            lines_list.append(word_crop)

            cv2.imshow(f'Line {i+1}', crops_ori[i])
            cv2.imwrite(f'output/Line No. {i+1}.png', crops_ori[i])

        print(f"More precisely there are: {word_count} words")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

       
        for i in range(len(lines_list)):
            cv2.imshow(f'Line {i+1}', crops_ori[i])
            for j in range(len(lines_list[i])):
                cv2.imshow(f'Word No. {len(lines_list[i])-j} out of {len(lines_list[i])} words present in the phrase No. {i+1}',lines_list[i][len(lines_list[i])-j-1])
                cv2.imwrite(f'output/Word No. {len(lines_list[i])-j} of phrase phrase No. {i+1}.png',lines_list[i][len(lines_list[i])-j-1])                
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return lines_list if len(lines) > 0 else print("There's nothing that seems as a useful text on the image") 
    
if __name__ == "__main__":
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    img = cv2.imread('output\warp.png')
    cv2.imshow('sdf',img)
    cv2.waitKey(0)
    seg_text(img)
