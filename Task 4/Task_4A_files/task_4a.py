from tensorflow import keras
import cv2
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Placeholder for utility functions
def utility_function_1():
   pass

def utility_function_2():
   pass

def task_4a_return():
   identified_labels = {"A": "", "B": "", "C": "", "D": "", "E": ""}

   model = keras.models.load_model("keras_model.h5", compile=False)
   class_names = open("labels.txt", "r").readlines()

   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "EDSR_x2.pb"
   sr.readModel(path)
   sr.setModel("edsr", 2)

   # Open webcam
   cap = cv2.VideoCapture(0)

   # Run the live feed for 5 seconds
   start_time = time.time()
   while time.time() - start_time < 7:
       ret, frame = cap.read()
       # Resize the live feed to the desired size
       frame = cv2.resize(frame, (920, 920), interpolation=cv2.INTER_AREA)
       cv2.imshow("Arena Feed", frame)
       cv2.waitKey(1)

   # Close the arena feed
   cv2.destroyAllWindows()

   # Capture a single frame
   ret, frame = cap.read()

   # Set the desired display size
   display_width = 920
   display_height = 920

   # Resize the raw image to the desired display size
   frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

   # # Coordinates and dimensions for 5 green boxes
   # box_coordinates = [(303, 767), (636, 589), (634, 413), (307, 405), (320, 143)]
   # box_width, box_height = 70, 75
   # box_w, box_h = 70, 69

      # Coordinates and dimensions for 5 green boxes
   box_coordinates = [(295, 769), (614, 591), (609, 416), (296, 408), (317, 146)]
   box_width, box_height = 60, 75
   box_w, box_h = 70, 69

   for (x, y) in box_coordinates:
       x1, y1 = x, y
       x2, y2 = x + box_width, y + box_height
       frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

   # Loop through each box and perform object detection on the captured frame
   for i, (x, y) in enumerate(box_coordinates):
       roi = frame[y:y + box_h, x:x + box_w]
       enhanced_roi = sr.upsample(roi)
       input_shape = (224, 224)
       enhanced_roi = cv2.resize(enhanced_roi, input_shape, interpolation=cv2.INTER_AREA)
       enhanced_roi_array = np.asarray(enhanced_roi, dtype=np.float32).reshape(1, 224, 224, 3)
       enhanced_roi_array = (enhanced_roi_array / 127.5) - 1
       prediction = model.predict(enhanced_roi_array)
       index = np.argmax(prediction)
       class_name = class_names[index].rstrip()
       confidence_score = prediction[0][index]
       label_text = f"Class: {class_name}"
       cv2.putText(frame, label_text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                  cv2.LINE_AA)
       identified_labels[chr(ord('A') + i)] = class_name

   # Display the identified labels
   print("identified_labels =", identified_labels)

   # Show the modified image with green boxes and object detection results
   cv2.imshow("Arena Feed", frame)

   # Wait for a key press and close the window
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   return identified_labels

if __name__ == "__main__":
   identified_labels = task_4a_return()