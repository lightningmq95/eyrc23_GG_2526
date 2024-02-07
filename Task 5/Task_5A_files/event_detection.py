from tensorflow import keras
import cv2
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def draw_and_measure_boxes(img, threshold_value=200, box_color=(0, 255, 0), box_thickness=2, min_area=6000, max_area=11000):
    # Convert to grayscale and threshold to create a binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # Find contours of the white boundaries
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store valid bounding boxes and areas
    bounding_boxes = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        area = w * h  # Calculate area

        # Filter boxes based on area range
        if min_area <= area <= max_area:
            # Draw the bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, box_thickness)
            bounding_boxes.append((x, y, w, h, area))

                # Display the image with bounding boxes and areas
    for i, box in enumerate(bounding_boxes):
        x, y, w, h, area = box
        cv2.putText(img, f"", (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    return bounding_boxes

def task_4a_return():
   identified_labels = {"A": "", "B": "", "C": "", "D": "", "E": ""}

   model = keras.models.load_model("keras_model.h5", compile=False)
   class_names = open("labels.txt", "r").readlines()

   sr = cv2.dnn_superres.DnnSuperResImpl_create()
   path = "EDSR_x2.pb"
   sr.readModel(path)
   sr.setModel("edsr", 2)

   # Load the image
   frame = cv2.imread(image_path)

#    # Open webcam
#    cap = cv2.VideoCapture(0)

#    # Run the live feed for 7 seconds
#    start_time = time.time()
#    while time.time() - start_time < 7:
#        ret, frame = cap.read()
#     #    # Resize the live feed to the desired size
#     #    frame = cv2.resize(frame, (920, 920), interpolation=cv2.INTER_AREA)
#        cv2.imshow("Arena Feed", frame)
#        cv2.waitKey(1)

#    # Close the arena feed
#    cv2.destroyAllWindows()

#    # Capture a single frame
#    ret, frame = cap.read()

# #    # Set the desired display size
# #    display_width = 920
# #    display_height = 920

# #    # Resize the raw image to the desired display size
# #    frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)

   # Use the boundary box detection function
   bounding_boxes = draw_and_measure_boxes(frame, min_area=6000, max_area=11000)

   for i, box in enumerate(bounding_boxes):
        x, y, w, h, _ = box
        roi = frame[y:(y - 5) + (h - 5), x:(x - 10) + (w - 10)]
        enhanced_roi = sr.upsample(roi)
        input_shape = (224, 224)
        enhanced_roi = cv2.resize(enhanced_roi, input_shape, interpolation=cv2.INTER_AREA)
        enhanced_roi_array = np.asarray(enhanced_roi, dtype=np.float32).reshape(1, 224, 224, 3)
        enhanced_roi_array = (enhanced_roi_array / 127.5) - 1
        prediction = model.predict(enhanced_roi_array)
        index = np.argmax(prediction)
        class_name = class_names[index].rstrip()
        confidence_score = prediction[0][index]
        detected_class = '' if class_name.lower() == 'unknown' else class_name
        cv2.putText(frame, f"Class: {detected_class}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        identified_labels[chr(ord('A') + i)] = detected_class
    #    cv2.putText(frame, label_text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    #    identified_labels[chr(ord('A') + i)] = class_name

   # Display the identified labels
   print("identified_labels =", identified_labels)

   # Show the modified image with green boxes and object detection results
   cv2.imshow("Arena Feed", frame)
   # Wait for a key press and close the window
   cv2.waitKey(0)
   cv2.destroyAllWindows()

   return identified_labels

if __name__ == "__main__":
   image_path = "Images/flex5.jpg"
   identified_labels = task_4a_return()