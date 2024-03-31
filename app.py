import easyocr
from pylab import rcParams
from PIL import Image
import streamlit as st
import numpy as np
from googletrans import Translator
from gtts import gTTS
import cv2

#preprocessing 
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarization(img):
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    return img

def noise_removal(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1) 
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    return img

def thin_font(img):
    img = cv2.bitwise_not(img)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.bitwise_not(img)
    return img

def prepro(img_path):
    img = cv2.imread(img_path)
    gray_img = grayscale(img)
    cv2.imwrite(img_path, gray_img)
    img = cv2.imread(img_path)
    bin_img = binarization(img)
    cv2.imwrite(img_path, bin_img)
    img = cv2.imread(img_path)
    nonoise_img = noise_removal(img)
    cv2.imwrite(img_path, nonoise_img)
    img = cv2.imread(img_path)
    thin_img = thin_font(img)
    cv2.imwrite(img_path, thin_img)

def ocr(img):
    rcParams['figure.figsize'] = 8, 16
    reader = easyocr.Reader(['en', 'hi', 'mr'])

    img = np.array(img)

    output = reader.readtext(img)

    result = ''
    for lists in output:
        result = result + " " + lists[-2]
    
    return result

def text_to_speech(text, audio_path):
    tts = gTTS(text=text)
    tts.save(audio_path)
    return audio_path

def translation(text, dest):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest)
    return translated_text.text

def main(rerunable):
    st.title("OCR based Language Translator")

    st.sidebar.write("Upload an image by clicking on the 'Browse files' button")
    st.sidebar.write("'Browse files' बटन पर क्लिक करके एक इमेज अपलोड करें")
    st.sidebar.write("'Browse files' बटणावर क्लिक करून इमेज अपलोड करा")
    # Upload an image for OCR
    img = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility='hidden')

    selected_language = st.sidebar.selectbox("Select Language", ['English', 'हिंदी', 'मराठी'])
    if selected_language == 'English':
        lang = 'en'
    elif selected_language == 'हिंदी':
        lang = 'hi'
    elif selected_language == 'मराठी':
        lang = 'mr'

    if img and selected_language:
        if rerunable == True:
            st.rerun()
        st.image(img, caption="Uploaded Image", width=200)
        performing_placeholder = st.empty()
        performing_placeholder.text("Performing OCR...")
        img_path = "image/img.jpg"
        img = Image.open(img)
        img = img.save(img_path)
        # prepro(img_path)
        img = cv2.imread(img_path)

        text = ocr(img)

        if text:
            trans_text = translation(text, lang)
            
            st.subheader('Original Text')
            st.write(text)

            st.subheader('Translated Text')
            st.write(trans_text)

            audio_path = text_to_speech(trans_text, "audio/tts.mp3")
            st.audio(audio_path, format="audio/mp3")
            rerunable = True
        else:
            st.warning("No text detected in the image.")

if __name__ == "__main__":
    main(rerunable=False)