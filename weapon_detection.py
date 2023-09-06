import cv2
import numpy as np
import pyrebase
from time import sleep

config = {
  "apiKey": "AIzaSyCYSrmGY-LqsSsyyjehJXVlX10VQVuEKLs",
  "authDomain": "home-security-7f99a.firebaseapp.com",
  "projectId": "home-security-7f99a",
  "databaseURL": "https://home-security-7f99a-default-rtdb.firebaseio.com",
  "storageBucket": "home-security-7f99a.appspot.com",
  "messagingSenderId": "674787418238",
  "appId": "1:674787418238:web:fe48bda8bab8f443e7382f",
  "databaseURL": "https://home-security-7f99a-default-rtdb.firebaseio.com"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
db.child("WeaponDetected").set(False)
storage = firebase.storage()
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# def value():
#     val = input("Enter file name or press enter to start webcam : \n")
#     if val == "":
#         val = 0
#     return val


# for video capture
cap = cv2.VideoCapture(0)
weapon_detected = False
print("Waiting for Motion...")

while True:
    if db.child("Motion").get().val() == True:
        _, img = cap.read()
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.000000000000000000000000000000000000000000000000002:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        if indexes == 0 or indexes == 1: 
            print("weapon has been detected!")
            if weapon_detected == False:
                db.child("weapon_trig").set(True)
                db.child("WeaponDetected").set(True)
                weapon_detected = True
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if weapon_detected == True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            img_name = "img.png"
            cv2.imwrite(img_name, img)
            print("image taken!")
            storage.child(img_name).put(img_name)
            print("image uploaded to database successfully!")
            break
cap.release()
cv2.destroyAllWindows()
sleep(20)
db.child("weapon_trig").set(False)
