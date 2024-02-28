"""
* Team Id: GG_2526
* Author List: Harshal Kale, Akash Mohapatra, Sharanya Anil
* Filename: image_classification.py
* Theme: GeoGuide
* Functions: classify, draw_and_measure_boxes, priority_list
* Global Variables: model, classes, image_transforms, label_mapping
"""

import cv2
import time
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np


# Load the model (Model has been uploaded on the drive along with its weights)
model = torch.load('model_newdv2l.pth')


# Define the classes
classes = ['combat', 'destroyed_buildings', 'fire', 'humanitarian_aid', 'military_vehicles']


# Define the image transforms
image_transforms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


'''
* Function Name: classify
* Input: model (torch.nn.Module) - The pre-trained neural network model.
         image_transforms - The image transformations to be applied.
         image - The input image to be classified.
         classes - The list of classes corresponding to the output labels of the model.
* Output: The predicted class label for the input images.
* Logic: Classifies the input image using the provided pytorch model trained on pretrained dataset efficientnet_v2s and returns the predicted class label.
* Example Call: classify(model, image_transforms, image, classes)
'''

def classify(model, image_transforms, image, classes):
    model = model.eval()
    # Move model to the same device as the image tensor
    device = next(model.parameters()).device
    # Apply transformations and move image to the device
    image = image_transforms(image).float().to(device)
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    return classes[predicted.item()]


'''
* Function Name: draw_and_measure_boxes
* Input: img - The input image as a NumPy array.
         threshold_value - The threshold value for binarization of the image.
         box_color - The color of the bounding boxes.
         box_thickness - The thickness of the bounding boxes.
         min_area - The minimum area of a contour to be considered for bounding box measurement.
         max_area - The maximum area of a contour to be considered for bounding box measurement.
* Output: bounding_boxes, img_with_boxes, identified_labels (dictionary)
* Logic: Detects and measures bounding boxes around objects in the input image, and classifies them.
         It returns the bounding boxes' information, the input image with drawn bounding boxes and labels,
         and the identified labels.
* Example Call: draw_and_measure_boxes(img)
'''

def draw_and_measure_boxes(img, threshold_value=200, box_color=(0,  255,  0), box_thickness=2, min_area=5000, max_area=13000):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, threshold_value,  255, cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    identified_labels = {"A": "", "B": "", "C": "", "D": "", "E": ""}
    
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        area = w * h

        if min_area <= area <= max_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, box_thickness)
            bounding_boxes.append((x, y, w, h, area))
    
    for i, box in enumerate(bounding_boxes):
        x, y, w, h, area = box
        cropped_img = img[y:y+h, x:x+w]
        label = classify(model, image_transforms, Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)), classes)
        identified_labels[chr(65 + i)] = label

        # Add white background for label
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,  0.9,  2)
        text_width, text_height = text_size
        cv2.rectangle(img, (x +  5, y -  10 - text_height), (x +  5 + text_width, y -  10), (255,  255,  255), -1)

        cv2.putText(img, label, (x +  5, y -  10), cv2.FONT_HERSHEY_SIMPLEX,  0.9, box_color,  2)

    return bounding_boxes, img, identified_labels


'''
* Function Name: priority_list
* Input: identified_labels - A dictionary containing identified labels for each bounding box.
* Output: Writes a priority list of labels to a text file.
* Logic: Generates a priority list of labels based on a predefined order and writes it to a text file.
* Example Call: priority_list(identified_labels)
'''

def priority_list(identified_labels):
    priority_order = ['fire', 'destroyed_buildings', 'humanitarian_aid', 'military_vehicles', 'combat']
    priority_list = []
    # Iterate through the priority order
    for label in priority_order:
        # Check if the label exists in the identified labels
        for key, value in identified_labels.items():
            if value.lower() == label:
                priority_list.append(f"{key},")
    # Write the priority list to a text file
    with open("priority_labels.txt", "w") as file:
        file.write("".join(priority_list).rstrip(','))


# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Capture a single frame after 5 seconds
time.sleep(5)
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not capture frame.")
    exit()

# Resize the frame to 980 by 1000
frame = cv2.resize(frame, (980, 1000))


# Call the function draw_and_measure_boxes to detect and measure bounding boxes around objects in the frame,
# and obtain the bounding box information, the image with drawn bounding boxes and labels and the identified labels.
bounding_boxes, img_with_boxes, identified_labels = draw_and_measure_boxes(frame)


# Mapping labels to more descriptive way to print on the terminal
label_mapping = {
    'fire': 'Fire',
    'humanitarian_aid': 'Humanitarian Aid and rehabilitation',
    'military_vehicles': 'Military Vehicles',
    'destroyed_buildings': 'Destroyed Buildings',
    'combat': 'Combat'
}

# Create a new dictionary with mapped labels
mapped_labels_dict = {}
for key, value in identified_labels.items():
    if value.lower() in label_mapping:
        mapped_labels_dict[key] = label_mapping[value.lower()]
    else:
        mapped_labels_dict[key] = ''

# Print the dictionary
print(mapped_labels_dict)


# Call the priority list function with identified_labels as its parameter
priority_list(identified_labels)


# Display the image with bounding boxes
cv2.imshow('Arena Feed with Image Classification', img_with_boxes)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()