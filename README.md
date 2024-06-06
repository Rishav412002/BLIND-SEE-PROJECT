import numpy as np
import cv2
import time
import pyttsx3

engine = pyttsx3.init()

def find_distance_from_bbox(object_height, bbox_height, focal_length):
    D = (object_height * focal_length) / bbox_height
    return D

def determine_direction(bounding_boxes):
    left_detected = any(box[0] < 320 for box in bounding_boxes)
    right_detected = any(box[0] + box[2] > 320 for box in bounding_boxes)

    direction = ""

    if left_detected and not right_detected:
        direction = "Move Right"
    elif right_detected and not left_detected:
        direction = "Move Left"
    elif left_detected and right_detected:
        direction = "Move Anywhere"
    else:
        direction = "Move Straight"

    return direction

# Opening camera for capturing frames     
video = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video.isOpened():
    print("Failed to open the camera")
    exit()

# Preparing variables for spatial dimensions of the frames
h, w = None, None

# Loading COCO class labels from file and Opening file
with open(r"C:\Users\RISHAV PRAJAPATI\OneDrive\Desktop\coco.names") as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet(r"C:\Users\RISHAV PRAJAPATI\OneDrive\Desktop\yolov3.cfg",
                                     r"C:\Users\RISHAV PRAJAPATI\OneDrive\Desktop\yolov3.weights")

layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[int(i) - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5
threshold = 0.3
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

recognized_persons = {}
f = 0
t = 0
unrecognized_frames_threshold = 30

start_time_object_detection = time.time()
start_time_speak = time.time()

while True:
    ret, frame = video.read()
    if not ret:
        break

    if w is None or h is None:
        h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    f += 1
    t += end - start

    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    bounding_boxes = []
    confidences = []
    classIDs = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classIDs.append(class_current)

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    if time.time() - start_time_object_detection >= 2:
        start_time_object_detection = time.time()

        direction = determine_direction(bounding_boxes)
        print("Direction:", direction)

        for i in results:  # Access the first element of the tuple inside the results array
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[classIDs[i]].tolist()

            label_text = "{}: {:.4f}".format(labels[int(classIDs[i])], confidences[i])

            object_height = 0.2
            bbox_height = box_height
            focal_length = 1000

            distance = find_distance_from_bbox(object_height, bbox_height, focal_length)

            if labels[int(classIDs[i])] not in recognized_persons or recognized_persons[labels[int(classIDs[i])]]['unrecognized_frames'] > unrecognized_frames_threshold:
                recognized_persons[labels[int(classIDs[i])]] = {
                    'distance': distance,
                    'unrecognized_frames': 0
                }

                label_text = "{} | Distance: {:.2f} meters".format(label_text, distance)

                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                cv2.putText(frame, "{:.2f}m".format(distance), (x_min, y_min - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                cv2.putText(frame, label_text, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                voice = labels[int(classIDs[i])] + " detected. " + label_text.split('|')[1].strip()
                engine.say(voice)
                engine.runAndWait()
            else:
                recognized_persons[labels[int(classIDs[i])]]['unrecognized_frames'] += 1

    if time.time() - start_time_speak >= 5:
        start_time_speak = time.time()

        direction = determine_direction(bounding_boxes)
        print("Direction:", direction)

        engine.say(direction)
        engine.runAndWait()

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
# BLIND-SEE-PROJECT
