import subprocess
import time
import pyautogui
import pyperclip
import os
import gtts
import pygetwindow as gw
from playsound import playsound
import pyttsx3



def NLP(text):
    '''
    NOTE: this code is a scrapping code that basically open a browser which is Google Chrome and get to chatgpt and ask to recorrect the extracted phrase and get back the answer
    The execution will depend on your internet speed, and device speed.
    Adjust the time.sleep() function according to your needs dont touch anything while this code is running please.
    You will know when this function is when when you see the browser screen bein closed

    THIS CODE IS MAINLY DONE FOR WINDOWS CHANGE WHAT IS NEEDED TO BE CHANGED IN THE COMMANDS
    '''

    # Open Chrome, change these if needed #
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"  # Adjust the path based on your Chrome installation #
    subprocess.Popen([chrome_path, '--window-size=1,1', '--new-window', 'https://chat.openai.com'])

    # Wait for Chrome to open #
    time.sleep(2)
    
    # COMMANDES #
    pyautogui.typewrite(f'Can you correct me this phrase in french and use ponctuation only if its necessary and only give me the answer do not add anything else to it: {text}')
    pyautogui.press('tab')
    pyautogui.press('enter')
    time.sleep(5)
    pyautogui.hotkey('ctrl', 'shift', 'i')
    time.sleep(2)
    
    pyautogui.hotkey('ctrl', 'f')
    pyautogui.typewrite('<p>')
    time.sleep(0.2)
    pyautogui.press('tab')
    pyautogui.press('tab')
    time.sleep(0.2)
    pyautogui.press('right')
    pyautogui.press('right')
    pyautogui.press('right')
    pyautogui.press('right')
    pyautogui.press('right')
    pyautogui.press('right')
    time.sleep(0.2)
    pyautogui.press('tab')
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(1)
    pyautogui.hotkey('alt', 'f4')
    content = pyperclip.paste().splitlines()
    for c in content:
        if c.startswith('StaticText'):
            print("Corrected phrase :", c[len('StaticText'):])
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[2].id)
            volume = engine.getProperty('volume')  
            engine.setProperty('volume',0.5)
            rate = engine.getProperty('rate')
            engine.setProperty('rate', 140)
            engine.say(c[len('StaticText'):])
            engine.runAndWait()
    return c[len('StaticText'):]

if __name__ == "__main__":
    folder_name = "output"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # random text #
    text = "jo suis tres boau et aszi inteligen "
    # Call the function #
    parag = NLP(text)

    file_path = 'output/file.txt'

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write the text to the file
        file.write(parag)

    # Confirmation message
    print("Text saved successfully")

