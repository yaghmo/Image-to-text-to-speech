import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Load the image


def char_seg(img, max_width = 1080, max_height = 720):
    # this function pre process the word and cut some of the edges to remove the undesired rest of characters #
    img = cv2.resize(img, (max_width, max_height))
    chars = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    hist_x = np.sum(thresh,1)/255
    peak = np.argmax(hist_x)

    # the border cleaner #
    thresh[:,:50] = 0
    thresh[:,max_width-50:] = 0
    thresh[:50,:] = 0
    thresh_copy = thresh.copy()
    thresh[peak-50:,:] = 0

    hist_y = np.sum(thresh,0)/255
    zero_peaks_indices = np.argwhere(hist_y == 0).flatten()

    # getting the separation coords to separate chars #
    peaks_ranges = []
    peak_range = [zero_peaks_indices[0]]
    for i in range(1, len(zero_peaks_indices)):
        if zero_peaks_indices[i] == zero_peaks_indices[i-1] + 1:
            peak_range.append(zero_peaks_indices[i])
        else:
            peaks_ranges.append(peak_range)
            peak_range = [zero_peaks_indices[i]]
    peaks_ranges.append(peak_range)

    lowest_peak_indices = [int((peak_range[0] + peak_range[-1]) / 2) for peak_range in peaks_ranges]

    plt.figure()
    plt.plot(hist_y)
    plt.title('Character segmentation indices')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(lowest_peak_indices, hist_y[lowest_peak_indices], 'ro')

    plt.show()
    char_liste = []
    char_to_display = []
    
    for i in range(1, len(lowest_peak_indices)):
        char_liste.append(thresh_copy[:,lowest_peak_indices[i-1]:lowest_peak_indices[i]])
    for i in range(1, len(lowest_peak_indices)):
        char_to_display.append(chars[:,lowest_peak_indices[i-1]:lowest_peak_indices[i]])
    
    for i, indice in enumerate(lowest_peak_indices):
        img = cv2.line(img, [indice,0], [indice,max_height], color= (0,0,255), thickness=1)
    for i in range(len(lowest_peak_indices)-1):
        cv2.imshow(f'Charecter No. {i+1} ',char_to_display[i])
        cv2.imwrite(f'output/Charater No. {i+1}.png',char_liste[i])                
    cv2.imshow('Character segmentation',img)
    cv2.imwrite('output\Character segmentation.png',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return char_liste


if __name__ == "__main__":
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    image = cv2.imread('output\Word No. 1 of phrase phrase No. 2.png')
    thresh = char_seg(image)
    