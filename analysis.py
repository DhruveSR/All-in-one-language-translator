import Levenshtein

# Function to calculate Word Error Rate (WER)
def wer(expected, transcribed):
    expected = expected.split()
    transcribed = transcribed.split()
    return Levenshtein.distance(expected, transcribed) / len(expected)

translation_dataset = [
    #en
    {"expected": "I am fine", "transcribed": "I'm fine"},# मैं ठीक हूँ
    {"expected": "The sun is shining", "transcribed": "The sun is shining"}, #सूरज चमक रहा है
    {"expected": "he sings very well", "transcribed": "He is very good sings"}, #वह अच्छा गाता था
    {"expected": "My name is Diya", "transcribed": "My name is diya"}, #मेरा नाम दीया है
    {"expected": "I like to play", "transcribed": "I like to play"}, #मुझे खेलना पसंद है
    #hi
    {"expected": "चटाई पर बिल्ली", "transcribed": "एक चटाई पर बिल्ली"}, # cat on a mat
    {"expected": "मुझे आराम की ज़रूरत है", "transcribed": "मुझे आराम की ज़रूरत है"}, #i need rest
    {"expected": "यहाँ बहुत ठंड है", "transcribed": " यहाँ बहुत ठंडा है"},#it is too cold here
    {"expected": "तुम कहां जा रहे हो", "transcribed": "आप कहां जा रहे हैं"}, #where are you going
    {"expected": "मैं भूखा हूँ", "transcribed": "मैं भूखा हूँ"}, #i am hungry
    #ma
    {"expected": "सूर्य चमकत आहे", "transcribed": "सूर्य चमकत आहे"}, #it is too cold here
    {"expected": "माझे नाव दिया आहे", "transcribed": "माझे नाव दिया आहे"},#my name is Diya
    {"expected": "हे पुस्तक माझे आहे", "transcribed": "हे पुस्तक माझे आहे"}, #this book is mine
    {"expected": "तू कसा आहेस", "transcribed": "तू कसा आहेस"}, #how are you
    {"expected": "मी प्रेमात आहे", "transcribed": "मी प्रेमात आहे"}, #i am in love
]

# Dataset containing audio transcriptions
audio_dataset = [
    #en
    {"expected": "My name is diya", "transcribed": "my name is diya"},
    {"expected": "the sun is shining", "transcribed": "sun is shining"},
    {"expected": "he is a very good singer", "transcribed": "he is very good singer"},
    {"expected": "there is a valley of flowers", "transcribed": "there is a valley of flowers"},
    {"expected": "I am fine", "transcribed": "I am fine"},
    #hi
    {"expected": "हरा मेरा पसंदीदा रंग है", "transcribed": "हर मेरा पसंदीदा रंग है"},
    {"expected": "मुझे बिल्लियां बहुत पसंद है", "transcribed": "मुझे बिल्लियां बहुत पसंद है"},
    {"expected": "मुझे भूख लगी है", "transcribed": "भूख लगी है"},
    {"expected": "तुम कहां जा रहे हो", "transcribed": "तुम कहां जा रहे हो"},
    {"expected": "मैं ठीक हूँ", "transcribed": "मैं ठीक हूं"},
    #ma
    {"expected": "हे पुस्तक माझे आहे", "transcribed": "हे पुस्तक माझे आहे"},
    {"expected": "तुम्ही कसे आहात", "transcribed": "तुम्ही कसे आहात"},
    {"expected": "मी खुप आनंद वाटतो", "transcribed": "मी खूप आनंद वाटतो"},
    {"expected": "माझे नाव दिया आहे", "transcribed": "माझे नाव दिया आहे"},
    {"expected": "सूर्य चमकत आहे", "transcribed": "सूर्य चमकत आहे"},
]

ocr_dataset = [
    #en
    {"expected": "Explain that stuff ! 01234567890", "transcribed": "Exp lain that ctuf f ! Dl२३4 ५८ ७८६०"},
    {"expected": "I'm a sample test", "transcribed": "I'm a sample test"},
    {"expected": "Noisy image to test Tesseract OCR", "transcribed": "Noisy image to test Tesseract OCR"},
    {"expected": "OCR", "transcribed": "OCR"},
    {"expected": "Tesseract sample", "transcribed": "Tesseract sample"},
    #hi
    {"expected": "ओसीआर", "transcribed": "ओसीआर"},
    {"expected": "नमस्ते", "transcribed": "नमस्ते"},
    {"expected": "हिंदी", "transcribed": "हिंद"},
    {"expected": "डेनमार्क मे एक ऐसा ही", "transcribed": "डेनमार्क मे एक ऐसा ही"},
    {"expected": "कावेरी ज्वैलर्स", "transcribed": "कावेर ज्वैलर्स"},
    #ma
    {"expected": "मराटेसाहा चाय स्पााॅत", "transcribed": "घशहिशाद्वि"}, 
    {"expected": "आपत्कालीन मार्ग", "transcribed": "आपत्कालीन मार्ग"},
    {"expected": "प्रवेश निषिद्ध", "transcribed": "प्रवेश निषिद्ध"},
    {"expected": "कार्यालय", "transcribed": "कार्यालय"},
    {"expected": "धोका अच्य विद्युत दाब दूर रहा", "transcribed": "धोका अच्य विद्युत दाब * दूर रहा"},
]

def calculate_wer(dataset):
    # Initialize variable to accumulate WER
    total_wer = 0

    # Calculate WER for each transcription and accumulate
    for i, data in enumerate(dataset):
        expected = data["expected"]
        transcribed = data["transcribed"]
        error_rate = wer(expected, transcribed)
        total_wer += error_rate
        print(f"check {i+1} - WER: {error_rate:.2f}")

    # Calculate average WER
    avg_wer = total_wer / len(dataset)
    return avg_wer

print(f"Average WER for Translation: {calculate_wer(translation_dataset):.3f}\n\n")
print(f"Average WER for Speech Recognition: {calculate_wer(audio_dataset):.3f}\n\n")
print(f"Average WER for OCR: {calculate_wer(ocr_dataset):.3f}\n\n")