# Some kind of license?
#
#
#
###############################################################################
## Imports libraries
import os
import numpy as np

###############################################################################
import packages.source.classes as cls        ## Imports included classes package
import packages.source.functions as fnc      ## Imports included functions package
###############################################################################
## Defines constants used
directory_working = os.path.dirname(os.path.realpath(__file__)) ## Gets the working directory of this file
directory_dataset = "F:\\Dataset\\Working Dataset" ## Defines the directory containing the dataset (must have '\\' between folders)

video_files = [] ## USed to store paths to video files
for root, dirs, files in os.walk(directory_dataset):	## Walks through provided directory and sub-directories
    for file in files:	## For every file in the current directory
        if file.lower().endswith(".mp4"): ## if an '.mp4' file
             video_files.append([root, file])	## Add to list
os.chdir(directory_working) ## Ensures back in working directory at end of walk

for i in range(len(video_files)):	## For every video file found
	file_path = video_files[i][0] + "\\" + video_files[i][1]	## Gets the file path
	print("\nExtracting ROI (" + str(i+1) + "/" + str(len(video_files)) + "): " + file_path.replace(directory_dataset, "Dataset"))
	output = fnc.read_roi(file_path) ## Check if the roi has already been extracted for this file
	if output[0] == 1: ## If file has been extracted previously
		print("ROI Extraction Already Present (" + str(i+1) + "/" + str(len(video_files)) + "): " + file_path.replace(directory_dataset, "Dataset"))
	else:
		fnc.extract_roi(file_path, display_roi = False) ## Extract the ROI for this source file, will write to same directory as source video after completion
		print("\nROI Extracted (" + str(i+1) + "/" + str(len(video_files)) + "): " + file_path.replace(directory_dataset, "Dataset"))

