import easyocr
from PIL import Image
import streamlit as st
import numpy as np
from googletrans import Translator
from gtts import gTTS
import cv2
import os
import speech_recognition as sr

def create_directories():
    if not os.path.exists("image"):
        os.makedirs("image")
    if not os.path.exists("audio"):
        os.makedirs("audio")

# Preprocessing 
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

def main():
    create_directories()
    st.title("Language Translator")

    st.sidebar.write("Choose Action:")
    action = st.sidebar.radio("", ('OCR', 'Translate Audio', 'Translate Text'))

    if action == 'OCR':
        selected_language = st.sidebar.multiselect("Select Language", ['English', 'हिंदी', 'मराठी'])
        lang_codes = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr'}
        
        if selected_language:
            lang = [lang_codes[lang] for lang in selected_language if lang in lang_codes]

            option = st.sidebar.radio("Select Input", ('Upload Image', 'Capture Image'))
            
            if option == 'Upload Image':
                
                st.sidebar.write("Upload an image by clicking on the 'Browse files' button")
                img = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility='hidden')

                if img:
                    st.image(img, caption="Uploaded Image", width=200)
                    performing_placeholder = st.empty()
                    performing_placeholder.text("Performing OCR...")
                    img_path = "image/img.jpg"
                    img = Image.open(img)
                    img = img.save(img_path)

                    #prepro(img_path)

                    img = cv2.imread(img_path)

                    text = ocr(img)

                    if text:
                        for l in lang:
                            trans_text = translation(text, l)

                            st.subheader(f'Translated Text ({l})')
                            st.write(trans_text)

                            audio_path = text_to_speech(trans_text, f"audio/tts_{l}.mp3")
                            st.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning("No text detected in the image.")
    
            elif option == 'Capture Image':
                img_file_buffer = st.camera_input("Capture Image")

                if img_file_buffer:

                    performing_placeholder = st.empty()
                    performing_placeholder.text("Performing OCR...")

                    bytes_data = img_file_buffer.getvalue() 
                    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

                    img_path = "image/img.jpg"
                    cv2.imwrite(img_path, img)

                    #prepro(img_path)

                    img = cv2.imread(img_path)
                    text = ocr(img)

                    if text:
                        for l in lang:
                            trans_text = translation(text, l)

                            st.subheader(f'Translated Text ({l})')
                            st.write(trans_text)

                            audio_path = text_to_speech(trans_text, f"audio/tts_{l}.mp3")
                            st.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning("No text detected in the image.")

    elif action == 'Translate Audio':
        selected_language = st.sidebar.selectbox("Select Original Language", ['English', 'हिंदी', 'मराठी'])
        src_lang_codes = {'English': 'en-US', 'हिंदी': 'hi-IN', 'मराठी': 'mr-IN'}
        src_lang = src_lang_codes[selected_language]

        selected_language = st.sidebar.selectbox("Select Language to Translate", ['English', 'हिंदी', 'मराठी'])
        dest_lang_codes = {'English': 'en', 'हिंदी': 'hi', 'मराठी': 'mr'}
        dest_lang = dest_lang_codes[selected_language]
        
        st.sidebar.write("Click below to start recording audio:")
        if st.sidebar.button("Start Recording"):
            recording_placeholder = st.empty()
            recording_placeholder.text("Recording...")

            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.pause_threshold = 1
                audio = recognizer.listen(source) 
  
            try:
                recording_placeholder.empty()
                transcribed_text = recognizer.recognize_google(audio, language=src_lang)

                st.subheader('Transcribed Text')
                st.write(transcribed_text)

                translated_text = translation(transcribed_text, dest_lang)

                st.subheader('Translated Text')
                st.write(translated_text)

                tts_audio_path = text_to_speech(translated_text, "audio/tts.mp3")
                st.audio(tts_audio_path, format="audio/mp3")

            except Exception: 
                st.write("Error: Unable to transcribe audio.")

    elif action == 'Translate Text':
    
        languages = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
        target_language = st.selectbox("Select target language", list(languages.keys()))

        input_text = st.text_area("Enter text to translate")

        if st.button("Translate"):
            if input_text.strip() != "":
                translated_text = translation(input_text, languages[target_language])
                st.success("Translated text:")
                st.write(translated_text)

                audio_path = text_to_speech(translated_text, f"audio/tts_{translated_text}.mp3")
                st.audio(audio_path, format="audio/mp3")
            else:
                st.warning("Please enter text to translate")

if __name__ == "__main__":
    main()