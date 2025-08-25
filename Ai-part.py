import face_recognition
import cv2
import numpy as np
import requests
from datetime import datetime


video_capture = cv2.VideoCapture(0)

bill_image = face_recognition.load_image_file("C:/Users/Amin/Desktop/PythonProject/bill.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

Tom_image = face_recognition.load_image_file("Tom Hardy10_1215.jpg")
Tom_face_encoding = face_recognition.face_encodings(Tom_image)[0]

holland_image = face_recognition.load_image_file("Tom Holland66_4848.jpg")
holland_face_encoding = face_recognition.face_encodings(holland_image)[0]

henry_image = face_recognition.load_image_file("Henry Cavil26_1207.jpg")
henry_face_encoding = face_recognition.face_encodings(henry_image)[0]

amin_image = face_recognition.load_image_file("amin.jpg")
amin_face_encoding = face_recognition.face_encodings(amin_image)[0]

aziz_image = face_recognition.load_image_file("aziz.jpg")
aziz_face_encoding = face_recognition.face_encodings(aziz_image)[0]

andy_image = face_recognition.load_image_file("Andy Samberg0_429.jpg")
andy_face_encoding = face_recognition.face_encodings(andy_image)[0]




known_face_encodings = [
    bill_face_encoding,
    Tom_face_encoding,
    holland_face_encoding,
    henry_face_encoding,
    amin_face_encoding,
    aziz_face_encoding,
    andy_face_encoding,

]
known_face_names = [
    "bill",
    "Tom",
    "holland",
    "henry",
    "amin",
    "aziz",
    "andy",



]

id = {
    "bill": 0,
    "Tom": 1,
    "holland": 2,
    "henry": 3,
    "amin": 4,
    "aziz": 5,
    "andy": 6,


}
adm ={
"bill": 0,
    "Tom": 1,
    "amin": 4,
   }



ESP32_IP = "http://192.168.1.166/msg"

def send_to_esp32(name, account_type, user_id):

    time_now = datetime.now().strftime("%H:%M:%S")

    if name == "unknown":

        payload = f"{name},{time_now},none,0"
        requests.get(f"{ESP32_IP}/unknown", params={"data": payload})

    else:

        payload = f"{name},{time_now},{account_type},{user_id}"
        requests.get(f"{ESP32_IP}/open", params={"data": payload})



process_this_frame = True

while True:

    ret, frame = video_capture.read()

    if process_this_frame:

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # fun send
            account_type  = ""
            user_id= id.get(name)
            if name in adm:
                account_type = "admin"

            else:
                account_type = "regular"

         #   send_to_esp32(name, account_type, user_id)

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + " " + str(id.get(name, 0)), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print(id.get(name))
    cv2.imshow('vid', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

video_capture.release()
cv2.destroyAllWindows()
