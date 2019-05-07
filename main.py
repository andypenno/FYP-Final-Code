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

default_video_input = os.path.join(directory_working, "video.mp4")   ## Defines the default video file used         

###############################################################################
## Main code loop
close_program = False    ## When set to True, will exit the program at the end of the current program loop

while (close_program != True): ## Will loop until user selection '0' is chosen
    selection_valid = False  ## Defines holder variable used for user selection
    while(selection_valid != True): ## Will loop until a valid selection has been chosen
        print("\n\n")
        print("0:  Close Program")
        print("1:  Build List of Faces")
        print("2:  Extract ROI")
        print("3:  Display an Extracted ROI")
        print("4:  Extract rPPG Signal")
        print("5:  Compare PPG data")
        print("6:  Determine Heart Rate")
#        print("7:  ")
#        print("8:  ")
#        print("9:  ")
        
        user_selection = input("Please enter your selection (0-9):  ")

###############################################################################            
        if user_selection == '0':   ## If user chooses to close the program
            close_program = True    ## Allows the program to close
            selection_valid = True  ## Allows selection loop to exit
###############################################################################            
        elif user_selection == '1': ## If the user chooses to build a list of faces
            print("\nBuild list of faces from:")
            print("0:  Webcam Input")
            print("1:  Default Video File")
            print("2:  Dataset")
			
            source_valid = False  ## Defines holder variable used for user selection
            while(source_valid != True): ## Will loop until a valid selection has been chosen
                source_selection = input("Please select video source (0-2):  ")
                if source_selection == '0': ## If user selects '0'
                    list_of_faces, unidentified_frames, capture_source_info = fnc.build_list_of_faces(
                                                                                 source = 'webcam',
                                                                                 display_faces = True               
                                                                                )
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '1': ## If user selects '1'
                    list_of_faces, unidentified_frames, capture_source_info = fnc.build_list_of_faces(
                                                                                 source = default_video_input,
                                                                                 display_faces = True                
                                                                                )
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '2': ## If user selects '2'
                    folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
                    if (folder_path[0] == 1):
                        folder_path = folder_path[1]
                    else:
                        break

                    video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
                    if (video_source[0] == 1):
                        video_source = video_source[1]
                    else:
                        break
                    
                    list_of_faces, unidentified_frames, capture_source_info = fnc.build_list_of_faces(
                                                                                 source = video_source,
                                                                                 display_faces = True
                                                                                )
                    source_valid = True  ## Allows exit of while loop
                else:
                    print("Invalid Argument.")
                    source_valid = False
			
            selection_valid = True
###############################################################################            
        elif user_selection == '2':
            print("\nExtract ROI from:")
            print("0:  Webcam Input")
            print("1:  Default Video File")
            print("2:  Dataset")
			
            source_valid = False  ## Defines holder variable used for user selection
            while(source_valid != True): ## Will loop until a valid selection has been chosen
                source_selection = input("Please select video source (0-2):  ")
                if source_selection == '0': ## If user selects '0'
                    output, capture_source_info = fnc.extract_roi(source = 'webcam')
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '1': ## If user selects '1'
                    output, capture_source_info = fnc.extract_roi(source = default_video_input)
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '2': ## If user selects '2'
                    folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
                    if (folder_path[0] == 1):
                        folder_path = folder_path[1]
                    else:
                        break

                    video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
                    if (video_source[0] == 1):
                        video_source = video_source[1]
                    else:
                        break
                    
                    output, capture_source_info = fnc.extract_roi(source = video_source)
                    source_valid = True  ## Allows exit of while loop
                else:
                    print("Invalid Argument.")
                    source_valid = False
			
            selection_valid = True
###############################################################################            
        elif user_selection == '3':
            print("\nDisplay ROI from:")
            print("0:  Default Video File")
            print("1:  Dataset")
			
            source_valid = False  ## Defines holder variable used for user selection
            while(source_valid != True): ## Will loop until a valid selection has been chosen
                source_selection = input("Please select video source (0/1):  ")
                if source_selection == '0': ## If user selects '0'
                    output = fnc.read_roi(source = default_video_input) ## Check if the roi has already been extracted for this file
                    if output[0] == 1: ## If file has been extracted previously
                        roi_list = output[1]
                        capture_source_info = output[2]
                        fnc.display_roi_video(roi_list, capture_source_info)
                    else:
                        print("Error opening extracted video.")
                    
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '1': ## If user selects '1'
                    folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
                    if (folder_path[0] == 1):
                        folder_path = folder_path[1]
                    else:
                        break

                    video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
                    if (video_source[0] == 1):
                        video_source = video_source[1]
                    else:
                        break
                    
                    output = fnc.read_roi(source = video_source) ## Check if the roi has already been extracted for this file
                    if output[0] == 1: ## If file has been extracted previously
                        roi_list = output[1]
                        capture_source_info = output[2]
                        fnc.display_roi_video(roi_list, capture_source_info)
                    else:
                        print("Error opening extracted video.")
                    
                    
                    source_valid = True  ## Allows exit of while loop
                else:
                    print("Invalid Argument.")
                    source_valid = False

            selection_valid = True
###############################################################################            
        elif user_selection == '4':
            print("\nExtract rPPG Signal from:")
            print("0:  Webcam Input")
            print("1:  Default Video File")
            print("2:  Dataset")
			
            source_valid = False  ## Defines holder variable used for user selection
            while(source_valid != True): ## Will loop until a valid selection has been chosen
                source_selection = input("Please select video source (0-2):  ")
                if source_selection == '0': ## If user selects '0'
                    output, capture_source_info = fnc.extract_rppg(source = 'webcam')
                                        
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '1': ## If user selects '1'
                    output, capture_source_info = fnc.extract_rppg(source = default_video_input)
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '2': ## If user selects '2'
                    folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
                    if (folder_path[0] == 1):
                        folder_path = folder_path[1]
                    else:
                        break

                    video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
                    if (video_source[0] == 1):
                        video_source = video_source[1]
                    else:
                        break
                    
                    output, capture_source_info = fnc.extract_rppg(source = video_source)
                    
                    source_valid = True  ## Allows exit of while loop
                else:
                    print("Invalid Argument.")
                    source_valid = False
			
            selection_valid = True
###############################################################################            
        elif user_selection == '5':
            folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
            if (folder_path[0] == 1):
                folder_path = folder_path[1]
            else:
                selection_valid = True ## Exits to main loop
                break
            
            video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
            if (video_source[0] == 1):
                video_source = video_source[1]
            else:
                selection_valid = True ## Exits to main loop
                break
                    
            rppg_signals, capture_source_info = fnc.extract_rppg(source = video_source)
            output = fnc.compare_ppg(rppg_signals, capture_source_info)
            
            selection_valid = True
###############################################################################
        elif user_selection == '6':
            print("\nDetermine Heart Rate from:")
            print("0:  Webcam Input")
            print("1:  Default Video File")
            print("2:  Dataset")
			
            source_valid = False  ## Defines holder variable used for user selection
            while(source_valid != True): ## Will loop until a valid selection has been chosen
                source_selection = input("Please select video source (0-2):  ")
                if source_selection == '0': ## If user selects '0'
                    rppg_signals, capture_source_info = fnc.extract_rppg(source = 'webcam')
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '1': ## If user selects '1'
                    rppg_signals, capture_source_info = fnc.extract_rppg(source = default_video_input)
                    source_valid = True ## Allows exit of while loop
                elif source_selection == '2': ## If user selects '2'
                    folder_path = fnc.dataset_folder_selection(directory_dataset)  ## Allows user selection of which subfolder to move to in provided directory
                    if (folder_path[0] == 1):
                        folder_path = folder_path[1]
                    else:
                        break

                    video_source = fnc.dataset_source_selection(directory_dataset, folder_path)  ## Allows user selection of a video source in all subfolders of 'folder_path'                 
                    if (video_source[0] == 1):
                        video_source = video_source[1]
                    else:
                        break
                    rppg_signals, capture_source_info = fnc.extract_rppg(source = video_source)
                    source_valid = True  ## Allows exit of while loop
                else:
                    print("Invalid Argument.")
                    source_valid = False
            
            if (source_valid):
                hr_rPPG, hr_PPG = fnc.determine_heart_rate(rppg_signals, capture_source_info)
            
            selection_valid = True
###############################################################################
#        elif user_selection == '7':
#            
#            selection_valid = True
############################################################################### 
#        elif user_selection == '8':
#            
#            selection_valid = True
############################################################################### 
#        elif user_selection == '9':
#            
#            selection_valid = True
###############################################################################              
        else:
            print("Invalid Argument.")
            selection_valid = False
###############################################################################            
            
            
            
            
            
            