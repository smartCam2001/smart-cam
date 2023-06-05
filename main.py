# ************************************** ALL LIBRARIES ************************************** #

from ultralytics import YOLO
import cv2
import os
import threading
import face_recognition
import numpy as np
import speech_recognition as sr
import pyttsx3
from PIL import Image
from deepface import DeepFace
import dlib
import time
import pytesseract
from scipy.spatial import distance
import winsound
import requests
import re
import datetime
import pipwin
import pyaudio

# ************************************** ALL VARIABLES ************************************** #

video = cv2.VideoCapture(0)  # To access Camera
model = YOLO('detect_key.pt')  # Model of Detect KKey
model2 = YOLO('detect_glass.pt')  # Model of Detect Glass
model3 = YOLO('yolov8x.pt')  # Model of YoloV8x
model4 = YOLO("detect_face.pt")  # Model of Detect Face
model5 = YOLO("detect_money.pt")  # Model of Detect Money
list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book"]  # Classes of Yolo
list2 = ["key", "glass"]
list_money = [.25, .25, .5, .5, 1, 1, 10, 10, 100, 20, 200, 5, 50]  # Classes of Model of Money
path1 = 'dataset'  # Path of Images that will be encoding
images = []  # List consists all Images as name_Image.jpg
classNames = []  # List consists all name of Images
encodeListKnown = []  # List consists all encoding of Images
focal_length = []  # List consists all focal_Length of objects(Key,Glasses)
count = 0  # Equal to the number of predictes in the folder runs
measured_distance = 23  # The distance calculated to calculate the focal length
real_width3 = 3  # the real width is used in calculate the focal_length & Distance
distance1 = 0  # The variable in which the distance is stored
counter = 0  # It is used in to remove image0.txt from folder after yhe first one
j = 0  # as a counter used in save image that analyze it
engine = pyttsx3.init()
engine.setProperty("rate", 100)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# ************************************** ALL FUNCTIONS ************************************** #

# Find Key & Glass in Room
def get_LocationKeyOrGlass(num):
    global counter
    str4 = ""
    str5 = ""
    if counter != 0:
        if count == 0:
            if os.path.isfile(f"runs\\detect\\predict\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict2\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict2\\labels\\image0.txt")
        else:
            if os.path.isfile(f"runs\\detect\\predict{count}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count}\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt")
    counter += 1
    if (num == 0):
        print('k')
        # t9 = threading.Thread(target=execution_models , args=(num3))
        # t9.start()
        model.predict(source=img, show=True, save=True, save_txt=True)
    elif (num == 1):
        print('g')
        # t10 = threading.Thread(target=execution_models , args=(num3))
        # t10.start()
        model2.predict(source=img, show=True, save=True, save_txt=True)
    # t11 = threading.Thread(target=execution_models , args=(num3))
    # t11.start()
    model3.predict(source=img, show=True, save=True, save_txt=True)
    # model3.predict(source = img , show = True , save = True, save_txt = True)
    # if num3 == 0:
    #     t9.join()
    #     t11.join()
    # elif num3 == 1:
    #     t10.join()
    #     t11.join()
    # t9.join()
    # t11.join()
    if count == 0:
        path_txt = f"runs\\detect\\predict\\labels\\image0.txt"
        if os.path.isfile(path_txt):
            with open(path_txt) as f:
                keys = f.readlines()
                for key in keys:
                    key = key.split(" ")
                    width_in_rf = get_perimeter(path_txt, img)
                    distance1 = Distance_finder(focal_length[num], real_width3, width_in_rf)
                    steps = round(distance1 / 35)
                    str1 = get_Location(key, img)
                    str2 = list2[num] + str1[0]
                    path_txt_objects = f"runs\\detect\\predict2\\labels\\image0.txt"
                    if os.path.isfile(path_txt_objects):
                        with open(path_txt_objects) as f:
                            objects = f.readlines()
                            for object in objects:
                                object = object.split(" ")
                                str3 = get_Location(object, img)
                                x_min, y_min, x_max, y_max = get_points(object, img)
                                str4 = get_LocationKeyByObjects(str1, x_min, y_min, x_max, y_max, object[0])
                                if str4 != "None":
                                    str5 += str(str4) + str3[0]
                            talk(f"There is a {str2}, {str5} ,{steps} steps away")
                            str5 = ""
                    else:
                        talk(f"There is a {str2} ,{steps} steps away and ")
                        talk(f"No objects nearby {list2[num]}")
        else:
            talk(f"No {list2[num]} Here")
    else:
        path_txt = f"runs\\detect\\predict{count}\\labels\\image0.txt"
        if os.path.isfile(path_txt):
            with open(path_txt) as f:
                keys = f.readlines()
                for key in keys:
                    key = key.split(" ")
                    width_in_rf = get_perimeter(path_txt, img)
                    distance1 = Distance_finder(focal_length_key, real_width3, width_in_rf)
                    steps = round(distance1 / 35)
                    str1 = get_Location(key, img)
                    str2 = list2[num] + str1[0]
                    path_txt_objects = f"runs\\detect\\predict{count + 1}\\labels\\image0.txt"
                    if os.path.isfile(path_txt_objects):
                        with open(path_txt_objects) as f:
                            objects = f.readlines()
                            for object in objects:
                                object = object.split(" ")
                                str3 = get_Location(object, img)
                                x_min, y_min, x_max, y_max = get_points(object, img)
                                str4 = get_LocationKeyByObjects(str1, x_min, y_min, x_max, y_max, object[0])
                                if str(str4) != "None":
                                    str5 += str(str4) + str3[0]
                            talk(f"There is a {str2}, {str5} ,{steps} steps away")
                            str5 = ""
                    else:
                        talk(f"There is a {str2} ,{steps} steps away and ")
                        talk(f"No objects nearby {list2[num]}")
        else:
            talk(f"No {list2[num]} Here")


# Get Direction of Key & Glass
def get_Location(key, img):
    heigth, width = img.shape[:2]
    x_center = round(float(key[1]) * width)
    y_center = round(float(key[2]) * heigth)
    if (0 <= x_center < round(width / 3) and 0 <= y_center < round(heigth / 3)):
        return ((" in TopLeft"), x_center, y_center)
    elif (round(width / 3) <= x_center < round(width / 3) * 2 and 0 <= y_center < round(heigth / 3)):
        return ((" in Top"), x_center, y_center)
    elif (round(width / 3) * 2 <= x_center < width and 0 <= y_center < round(heigth / 3)):
        return ((" in TopRigth"), x_center, y_center)
    elif (0 <= x_center < round(width / 3) and round(heigth / 3) <= y_center < round(heigth / 3) * 2):
        return ((" in Left"), x_center, y_center)
    elif (round(width / 3) <= x_center < round(width / 3) * 2 and round(heigth / 3) <= y_center < round(
            heigth / 3) * 2):
        return ((" in Center"), x_center, y_center)
    elif (round(width / 3) * 2 <= x_center < width and round(heigth / 3) <= y_center < round(heigth / 3) * 2):
        return ((" in Right"), x_center, y_center)
    elif (0 <= x_center < round(width / 3) and round(heigth / 3) * 2 <= y_center < heigth):
        return ((" in BottomLeft"), x_center, y_center)
    elif (round(width / 3) <= x_center < round(width / 3) * 2 and round(heigth / 3) * 2 <= y_center < heigth):
        return ((" in Bottom"), x_center, y_center)
    elif (round(width / 3) * 2 <= x_center < width and round(heigth / 3) * 2 <= y_center < heigth):
        return ((" in BottomRigth"), x_center, y_center)

    # Calculate (x_min,y_min) & (x_max,y_max) Of Objects


def get_points(object, frame):
    heigth_frame, width_frame = frame.shape[:2]
    width = round(float(object[3]) * width_frame)
    height = round(float(object[4]) * heigth_frame)
    x_center = round(float(object[1]) * width_frame)
    y_center = round(float(object[2]) * heigth_frame)
    x_min = x_center - round(width / 2)
    x_max = x_center + round(width / 2)
    y_min = y_center - round(height / 2)
    y_max = y_center + round(height / 2)
    return x_min, y_min, x_max, y_max


# Location of Key & Glass By Objects
def get_LocationKeyByObjects(str3, x_min, y_min, x_max, y_max, cls):
    keyX_center = str3[1]
    keyY_center = str3[2]
    if (x_min <= keyX_center <= x_max and y_min <= keyY_center <= y_max):
        return (f" and the key on the {list[int(cls)]}")
    if (x_max < keyX_center < x_max + 200 and y_min < keyY_center < y_max):
        return (f" and the key on the rigth of {list[int(cls)]}")
    if (x_min - 200 < keyX_center < x_min and y_min < keyY_center < y_max):
        return (f" and the key on the left of {list[int(cls)]}")
    if (x_min < keyX_center < x_max and y_max < keyY_center < y_max + 250):
        return (f" and the key on the bottom {list[int(cls)]}")
    if (x_min < keyX_center < x_max and y_min - 250 < keyY_center < y_min):
        return (f" and the key on the top of {list[int(cls)]}")


# Calculate perimeter
def get_perimeter(path, img):
    with open(path) as f:
        content = f.readlines()
    for obj in content:
        line = obj.split(" ")
        real_height, real_width = img.shape[:2]
        width = round(float(line[3]) * real_width)
        heigth = round(float(line[4]) * real_height)
        width_in_rf_image = (width + heigth) / 2
    return width_in_rf_image


# Calaculate FocalLength
def FocalLength(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# Calculate Distance
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (Focal_Length * real_face_width) / face_width_in_frame
    return distance


# findEncoding() return encodeing of each image
def findEncoding(path1):
    personList = os.listdir(path1)
    for cl in personList:
        curPerson = cv2.imread(f'{path1}/{cl}')
        images.append(curPerson)
        classNames.append(os.path.splitext(cl)[0])
        img = cv2.cvtColor(curPerson, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeListKnown.append(encode)


# Instead of FaceLocation Of Lib face-recognize
def get_LocationFaces(faces, frame):
    heigth_frame, width_frame = frame.shape[:2]
    locationFaces = []
    for line in faces:
        face = line.split(" ")
        width = round(float(face[3]) * width_frame)
        height = round(float(face[4]) * heigth_frame)
        x_center = round(float(face[1]) * width_frame)
        y_center = round(float(face[2]) * heigth_frame)
        left = x_center - round(width / 2)
        rigth = x_center + round(width / 2)
        top = y_center - round(height / 2)
        bottom = y_center + round(height / 2)
        locationFaces.append((top, rigth, bottom, left))
    return locationFaces


# Face Recognizer
def getPeoples(num):
    global counter
    talk('f')
    if counter != 0:
        if count == 0:
            if os.path.isfile(f"runs\\detect\\predict\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict2\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict2\\labels\\image0.txt")
        else:
            if os.path.isfile(f"runs\\detect\\predict{count}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count}\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt")
    counter += 1
    model4.predict(source=img, show=True, save_txt=True, save=True)
    if count == 0:
        if os.path.isfile(f"runs\\detect\\predict\\labels\\image0.txt"):
            with open(f"runs\\detect\\predict\\labels\\image0.txt") as f:
                faces = f.readlines()
            list = get_LocationFaces(faces, img)
            cv2.rectangle(img, (list[0][3], list[0][0]), (list[0][1], list[0][2]), (255, 255, 255), 5, cv2.FILLED)
            cv2.imshow("circles", img)
            encodeCurrentFrame = face_recognition.face_encodings(img, list)
            for encodeFace, faceLoc in zip(encodeCurrentFrame, list):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.5)  # tolerance
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].lower()
                    y1, x2, y2, x1 = faceLoc
                    if num == 0:
                        talk(name)
                    elif num == 1:
                        sleep(name)
                    elif num == 2:
                        if name in recorder and "emotion" in recorder:
                            print("PIL")
                            pil_image = Image.fromarray(img[y1:y2, x1:x2])
                            pil_image.save(f'analyze/{name}{j}.jpg')
                            # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                            # b.start()
                            speak_emotion(f'analyze/{name}{j}.jpg', name)
                        if name in recorder and "gender" in recorder:
                            print("PIL")
                            pil_image = Image.fromarray(img[y1:y2, x1:x2])
                            pil_image.save(f'analyze/{name}{j}.jpg')
                            # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                            # b.start()
                            speak_gender(f'analyze/{name}{j}.jpg', name)
                        if name in recorder and "age" in recorder:
                            print("PIL")
                            pil_image = Image.fromarray(img[y1:y2, x1:x2])
                            pil_image.save(f'analyze/{name}{j}.jpg')
                            # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                            # b.start()
                            speak_age(f'analyze/{name}{j}.jpg', name)
                else:
                    name = "Unknown"
                    talk(name)
        else:
            talk("No people Here")
    if os.path.isfile(f"runs\\detect\\predict{count}\\labels\\image0.txt"):
        with open(f"runs\\detect\\predict{count}\\labels\\image0.txt") as f:
            faces = f.readlines()
        list = get_LocationFaces(faces, img)
        cv2.rectangle(img, (list[0][3], list[0][0]), (list[0][1], list[0][2]), (255, 255, 255), 5, cv2.FILLED)
        cv2.imshow("circles", img)
        encodeCurrentFrame = face_recognition.face_encodings(img, list)
        for encodeFace, faceLoc in zip(encodeCurrentFrame, list):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace, 0.5)  # tolerance
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].lower()
                y1, x2, y2, x1 = faceLoc
                print(name)
                if num == 0:
                    talk(name)
                elif num == 1:
                    sleep(name)
                elif num == 2:
                    print(name)
                    if name in recorder and "emotion" in recorder:
                        print("nmj")
                        pil_image = Image.fromarray(img[y1:y2, x1:x2])
                        pil_image.save(f'analyze/{name}{j}.jpg')
                        # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                        # b.start()
                        speak_emotion(f'analyze/{name}{j}.jpg', name)
                    if name in recorder and "gender" in recorder:
                        pil_image = Image.fromarray(img[y1:y2, x1:x2])
                        pil_image.save(f'analyze/{name}{j}.jpg')
                        # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                        # b.start()
                        speak_gender(f'analyze/{name}{j}.jpg', name)
                    if name in recorder and "age" in recorder:
                        pil_image = Image.fromarray(img[y1:y2, x1:x2])
                        pil_image.save(f'analyze/{name}{j}.jpg')
                        # b = threading.Thread(target=speak_emotion, args=(f'analyze/{name}{j}.jpg', name,))
                        # b.start()
                        speak_age(f'analyze/{name}{j}.jpg', name)
            else:
                name = "Unknown"
                talk(name)
    else:
        talk("No people Here")


# Calculte Amount Of Money
def calculate_Money():
    global counter
    count_money = 0
    if counter != 0:
        if count == 0:
            if os.path.isfile(f"runs\\detect\\predict\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict2\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict2\\labels\\image0.txt")
        else:
            if os.path.isfile(f"runs\\detect\\predict{count}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count}\\labels\\image0.txt")
            if os.path.isfile(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt"):
                os.remove(f"runs\\detect\\predict{count + 1}\\labels\\image0.txt")
    counter = counter + 1
    model5.predict(source=img, show=True, save=True, save_txt=True)
    if count == 0:
        path_txt = f"runs\\detect\\predict\\labels\\image0.txt"
        if os.path.isfile(path_txt):
            with open(path_txt) as f:
                moneys = f.readlines()
                for money in moneys:
                    money = money.split(" ")
                    count_money += list_money[int(money[0])]
        talk(str(count_money) + " Pounds")
    else:
        path_txt = f"runs\\detect\\predict{count}\\labels\\image0.txt"
        if os.path.isfile(path_txt):
            with open(path_txt) as f:
                moneys = f.readlines()
                for money in moneys:
                    money = money.split(" ")
                    count_money += list_money[int(money[0])]
        talk(str(count_money) + " Pounds")


# To Record the sound coming from the speaker
def rec_audio():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        ReadyChirp1()  # when you hear that , you recognize that the microphone is listening for your commond
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
        ReadyChirp2()  # that mean that the device recognized the commond
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)
        print("Say that again please...")
        return "None"

    return query


def ReadyChirp1():
    winsound.Beep(600, 300)  # this is for pitch and tone length if you want to change it .. change the n's


def ReadyChirp2():
    winsound.Beep(500, 300)


# To Talk with me
def talk(audio):
    engine.say(audio)
    engine.runAndWait()


# To speak Emtoion
def speak_emotion(image, name):
    global j
    face_analysis = DeepFace.analyze(image, enforce_detection=False)
    talk(name + face_analysis["dominant_emotion"])
    j = j + 1  # for save pil_image


# To speak age
def speak_age(image, name):
    global j
    face_analysis = DeepFace.analyze(image, enforce_detection=False)
    talk(name + str(face_analysis["age"]))
    j = j + 1  # for save pil_image


# To speak gender
def speak_gender(image, name):
    global j
    face_analysis = DeepFace.analyze(image, enforce_detection=False)
    talk(name + face_analysis["gender"])
    j = j + 1  # for save pil_image


# Calculate Ratio
def calculate_R(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ratio = (A + B) / (2.0 * C)
    return ratio


def speak_time():
    strTime = datetime.datetime.now().strftime("%H:%M:%S")
    talk(f"Sir, the time is {strTime}")


def speak_wheather():
    search = "temperature"

    url = f"http://www.google.com/search?q={search}"

    r = requests.get(url)

    # use regular expression to extract temperature from the response

    temperature = re.search('<div class="BNeawe iBp4i AP7Wnd">([\d]+).*?</div>', r.text)

    if temperature:

        talk(f"The current temperature is {temperature.group(1)} degrees Celsius.")

    else:

        talk("Sorry, I could not find the temperature.")


# Recognize Color
def rec_Color():
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = img.shape

    cx = int(width / 2)
    cy = int(height / 2)

    # Pick pixel value
    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]
    s_value = pixel_center[1]
    v_value = pixel_center[2]

    color = "Undefined"
    if hue_value < 5:
        color = "RED"
    elif s_value < 40:
        color = "White"
    elif v_value < 20:
        color = "Black"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 33:
        color = "YELLOW"
    elif hue_value < 78:
        color = "GREEN"
    elif hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "RED"

    talk(f"the color is {color}")


# To know someone is sleeping or not
def sleep(name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        left_eye_ratio = calculate_R(leftEye)
        right_eye_ratio = calculate_R(rightEye)

        RATIO = (left_eye_ratio + right_eye_ratio) / 2
        RATIO = round(RATIO, 2)

        if RATIO < 0.26:
            if time1 == 0:
                time1 = time.time()
            time2 = time.time()
            alert = wake_up_time(time1, time2)
            if alert:
                talk(f'{name} is Drowsy')
                time1 = 0
                alert = False
        else:
            time1 = 0
        print(RATIO)


# The time during which a person is asleep is determined
def wake_up_time(t1, t2):
    if (t2 - t1) > 9:
        return True
    return False


# To read any Text
def read_Book():
    txt = pytesseract.image_to_string(gray)
    talk(txt)


# Encoding for each Known image
print("Start Encoding")
t1 = threading.Thread(target=findEncoding, args=(path1,))
t1.start()
print("End Encoding")

# Calculate Focal Length of Key
path_txt_key = "image0.txt"
img_key = cv2.imread("image0.jpg")
width_in_rf_image_key = get_perimeter(path_txt_key, img_key)
focal_length_key = FocalLength(measured_distance, real_width3, width_in_rf_image_key)
focal_length.append(focal_length_key)

# Calculate Focal Length of glass
path_txt_glass = "image1.txt"
img_glass = cv2.imread("image1.jpg")
width_in_rf_image_glass = get_perimeter(path_txt_glass, img_glass)
focal_length_glass = FocalLength(measured_distance, real_width3, width_in_rf_image_glass)
focal_length.append(focal_length_glass)

# Calculate var count
for a in range(50):
    if count == 0:
        if os.path.isdir(f"runs\\detect\\predict") == True:
            count += 2
            continue
        else:
            break
    else:
        if os.path.isdir(f"runs\\detect\\predict{count}") == True:
            count += 1
            continue
        else:
            break

t1.join()
# Stream
while True:
    _, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("scsjc", img)
    pressed_key = cv2.waitKey(50)

    if pressed_key == ord('r') or pressed_key == ord('R'):
        print("R")
        recorder = rec_audio()
        print(recorder)
        if "get my key" in recorder:
            num = 0
            print("enter key")
            # t1 = threading.Thread(target=get_LocationKeyOrGlass, args=(num,))
            # t1.start()
            get_LocationKeyOrGlass(num)
        if "get my glasses" in recorder:
            print("enter glasses")
            num = 1
            # t2 = threading.Thread(target=get_LocationKeyOrGlass, args=(num,))
            # t2.start()
            get_LocationKeyOrGlass(num)
        if "who with me" in recorder or "who is me" in recorder:
            print("enter with me")
            num2 = 0
            # t3 = threading.Thread(target=getPeoples, args=(num2,))
            # t3.start()
            getPeoples(num2)
        if "Is anyone sleep" in recorder:
            print("enter slepping")
            num2 = 1
            t4 = threading.Thread(target=getPeoples, args=(num2,))
            t4.start()
        if ("emotion" in recorder and "abdul" in recorder) or ("age" in recorder and "abdul" in recorder) or (
                "gender" in recorder and "abdul" in recorder):
            print("enter emotion abdo")
            num2 = 2
            # t5 = threading.Thread(target=getPeoples, args=(num2,))
            # t5.start()
            getPeoples(num2)
        if ("emotion" in recorder and "abdul" in recorder) or ("age" in recorder and "abdul" in recorder) or (
                "gender" in recorder and "abdul" in recorder):
            print("enter emotion ahmed")
            num2 = 2
            # t6 = threading.Thread(target=getPeoples, args=(num2,))
            # t6.start()
            getPeoples(num2)
        if "How much money" in recorder:
            print("enter money")
            # t7 = threading.Thread(target=calculate_Money)
            # t7.start()
            calculate_Money()
        if "read the sheet" in recorder:
            print("enter read")
            # t8 = threading.Thread(target=read_Book)
            # t8.start()
            read_Book()
        if 'the time' in recorder:
            print("wait")
            speak_time()
        if 'the wheather' in recorder:
            print("wait")
            speak_wheather()
        if 'color' in recorder or 'colour' in recorder:
            print("wait")
            rec_Color()
        if "exit" in recorder:
            break
cv2.destroyAllWindows()
video.release()