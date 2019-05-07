# Some kind of license?
#
#
#
###############################################################################
## Imports libraries
import os
import numpy as np
import csv

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

data_file_path = os.path.join(directory_dataset, 'hr_data.csv')
with open(data_file_path, mode='w') as output_file:
    output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    output_writer.writerow(['Source', 'Frame-rate', 'rPPG HR', '', 'PPG HR'])
    for i in range(len(video_files)):	## For every video file found
        file_path = video_files[i][0] + "\\" + video_files[i][1]	## Gets the file path
        print("\nDetermining HR (" + str(i+1) + "/" + str(len(video_files)) + "): " + file_path.replace(directory_dataset, "Dataset") + "\n")
        success = False
        try:
            rppg_signals, capture_source_info = fnc.extract_rppg(source = file_path)
            hr_rPPG, hr_PPG = fnc.determine_heart_rate(rppg_signals, capture_source_info)
            success = True
        except:
            print("Failed to Determine HR")
            output_writer.writerow([file_path, 0, 0, 0, 0, 0])
        if (success == True):
            output_writer.writerow([file_path, capture_source_info.framerate, hr_rPPG[0], hr_rPPG[1], hr_PPG[0], hr_PPG[1]])