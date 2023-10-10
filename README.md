# Image-to-text-to-speech
Extract text from images, correct it using NLP, and synthesize it into audio.


# Overview
This project aims to extract text from images, even if the images contain a board or a large rectangular area. The program performs automatic region selection when text is detected within a rectangle. If no text is detected, manual region selection can be used. The extracted text is then processed to create coherent paragraphs.

# Key Features
Text extraction from images with automatic or manual region selection.
Paragraph region extraction based on vertical histograms.
Line determination within paragraphs.
Word segmentation for each phrase.
Character recognition using a Convolutional Neural Network (CNN) model.
Reformation of paragraphs from recognized characters.
Utilization of ChatGPT for Natural Language Processing (NLP) with web scraping capabilities.
Correction of phrases for improved readability and correctness.
Generation of audio output for the extracted and corrected text.


# Usage
Install the necessary dependencies listed in the requirements.txt file.
Run the program and provide an image containing text as input.
The program will automatically detect text regions within rectangles. If no text is detected, manual region selection can be performed.
The extracted text will be processed to create paragraphs, lines, and words.
Character recognition using the CNN model will transform words into recognized characters.
The recognized characters will be used to reformulate paragraphs.
NLP with ChatGPT will be employed for text correction and enhancement.
The final corrected text will be converted into audio output for user-friendly accessibility.
