'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2C of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ 2526 ]
# Author List:		[ Harshal Vilas Kale, Akash Mohapatra, Sharanya Anil, Soham Gujar ]
# Filename:			task_2c.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import cv2 as cv       # OpenCV Library
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import torch
from cv2 import dnn_superres

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
arena_path = "arena.png"            # Path of generated arena image
event_list = []
detected_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

# Extracting Events from Arena
def arena_image(arena_path):            # NOTE: This function has already been done for you, don't make any changes in it.
    ''' 
	Purpose:
	---
	This function will take the path of the generated image as input and 
    read the image specified by the path.
	
	Input Arguments:
	---
	`arena_path`: Generated image path i.e. arena_path (declared above) 	
	
	Returns:
	---
	`arena` : [ Numpy Array ]

	Example call:
	---
	arena = arena_image(arena_path)
	'''
    '''
    ADD YOUR CODE HERE
    '''
    frame = cv.imread(arena_path)
    arena = cv.resize(frame, (700, 700))
    return arena 

def event_identification(arena):
    ''' 
    Purpose:
    ---
    This function will select the events on the arena image and extract them as
    separate images. The extracted images will be resized to 50x50 pixels.
    
    Input Arguments:
    ---
    `arena`: Numpy array representing the arena image
    
    Returns:
    ---
    `event_list`: List
        event_list will store the extracted event images

    Example call:
    ---
    event_list = event_identification(arena)
    '''

    # Initialize an empty list to store the extracted event images
    event_list = []
    
    # Define the fixed positions for the 5 events
    event_positions = [
        (154,598,205,648),# Event 1 coordinates (x1, y1) to (x2, y2)
        (460,468,511,519),# Event 2 coordinates,
        (464,335,515,386),# Event 3 coordinates,
        (145,335,196,386),# Event 4 coordinates,
        (158,119,209,170)# Event 5 coordinates,
    ]

    # Loop through each event position and extract the corresponding region
    for (x1, y1, x2, y2) in event_positions:
        # Print the coordinates for debugging
        print(f"Coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        # Extract the event region from the arena
        event_image = arena[y1:y2, x1:x2]
        
        
        # Print the size of the extracted region
        print(f"Event image size before resizing: {event_image.shape}")
        
        # Resize the event image to 50x50 pixels
        event_image_resized = cv.resize(event_image, (50, 50))
        
                # Create an SR object
        sr = dnn_superres.DnnSuperResImpl_create()

        # Read the desired model
        path = "EDSR_x2.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 2)

        # Upscale the image
        result = sr.upsample(event_image_resized)

        # Append the resized event image to the event_list
        event_list.append(result)
        # cv.imshow("image", event_image_resized)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    return event_list

# Event Detection
def classify_event(image):
    # Load the PyTorch model
    model = torch.load('model.pth', map_location=torch.device('cpu'))

    model.eval()  # Set the model to evaluation mode
    # Preprocess the event image (resize, normalize, etc.) 
    image = cv.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = torch.from_numpy(image)  # Convert to PyTorch tensor
    image = image.permute(0, 3, 1, 2)  # Change tensor layout to NCHW
    # Perform forward pass to classify the event
    with torch.no_grad():
        output = model(image)
    probabilities = torch.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    #model = model.load('model.pth')
    class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
    event = class_names[predicted_class.item()]

    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''
###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(event_list):
    for img_index in range(0,5):
        img = event_list[img_index]
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    os.remove('arena.png')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)

def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    input_function()
    #################

    ##### Process #####
    arena = arena_image(arena_path)
    event_list = event_identification(arena)
    detected_list = classification(event_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('arena.png'):
            os.remove('arena.png')
        if os.path.exists('detected_events.txt'):
            os.remove('detected_events.txt')
        sys.exit()