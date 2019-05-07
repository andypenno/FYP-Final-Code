# Some kind of license?
#
#
#
#
###############################################################################
## Imports Libraries
import numpy as np
import cv2 as cv
import os
import shutil   ## Allows deletion of non-empty directories
from scipy import signal as sps
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.ioff() ## Turns interactive plotting off (allows plt figures to be created, saved and then closed without displaying)
from scipy.signal import butter, lfilter
import csv
import imutils
##import peakutils


###############################################################################
working_directory = os.getcwd()            ## Gets current working directory
packages_directory = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..')) ## Gets this file's path and goes 'back' two folders
os.chdir(packages_directory)               ## Change to the correct directory to import the classes package
import packages.source.classes as cls      ## Imports included classes package
import packages.source.peakdetect as peak  ## Imports peak detector by Eli Billauer
os.chdir(working_directory)                ## Changes back to working directory

directory_cascade_classifier = "C:\\ProgramData\\Anaconda3\\pkgs\\opencv-3.4.4-py37hb76ac4c_1204\\Library\\etc\\haarcascades" ## Defines the directory containing the haar cascade '.xml' files
default_cascade_xml = os.path.join(directory_cascade_classifier, "haarcascade_frontalface_alt2.xml") ## Defines the haar cascade used

###############################################################################
def save_graph_plot(plot_output_path, ## Directory to save the plot in
                    plot_name, ## Output filename of the plot, do not add file type
                    subplots,  ## Number of subplots in frame
                    plot_data, ## Expects list of arrays [plot_x_data (list) , plot_y_data (list) , plot_colour (e.g. 'k' or 'b')]
                    silent = False
                    ):
    fig_width = int(len(plot_data[0][0])/20), ## Width of the plot
    if (subplots <= 3): ## If only plotting up to 3 graphs
        fig_height = int(15 * subplots)  ## Defines height of the plot  
    else:
        fig_height = int(2 * subplots)  ## Defines height of the plot
    
    fig = plt.figure(figsize=((fig_width[0], fig_height))) ## Sets the size of the output plot, is set to extreme value to give finer detail in output image

    for i in range(subplots): ## For every sub-plot in 'plot_data'
        plt.subplot(subplots, 1, (i+1))  ## Sets up subplot
        plt.plot(plot_data[i][0], plot_data[i][1], plot_data[i][2]) ## Plots data

    
    image_name = plot_name + '.png' ## Adds file type to image name
    save_path = os.path.join(plot_output_path, image_name)  ## Defines the full directory path of the output   
    fig.savefig(save_path) ## Saves the output figure to a .png image file
    if (silent == False):
        print("\nPlot figure saved to: " + save_path)
        
    plt.close(fig)

    return 0

###############################################################################
def display_roi_video(output, capture_source_info):
    counter = [(0*i) for i in range(len(output))] ## Defines a list of counters to count through each subject list in output
    for j in range(capture_source_info.frames): ## For every frame passed in the source video
        display_area = np.array([]) ## Defines empty array for this frame
        for i in range(len(output)): ## For every subject in list
            temp = np.array([]) ## Defines empty array
            if (output[i][counter[i]].current_frame == j): ## If the frame matches the current j
                for k in range(2,-1,-1): ## For each colour channel, R,G,B in turn
                    img = resize_by_height(output[i][counter[i]].roi, height = 250) ## Ensures is height 200 for output
                    holder = np.zeros(img.shape, dtype = np.uint8) ## Creates an array of zeros the same size as image
                    holder[:,:,k] = img[:,:,k] ## Extracts the current colour channel
                    temp = np.vstack([temp, holder]) if temp.size else holder ## Adds colour channel to holder array
                display_area = np.hstack([display_area, temp]) if display_area.size else temp  ## Adds subject RGB components to display image  
                counter[i] += 1 ## Increments faces counter
                if (counter[i] >= len(output[i])): ## Stops overflowing list
                    counter[i] -= 1
        if display_area.size: ## If there is anything to display
            cv.imshow('Extracted ROI', display_area) ## Displays all ROI's for current frame
            if cv.waitKey(10) & 0xFF == ord('q'):  ## Waits 10ms and allows to break out of video by pressing q
                break    
    cv.destroyAllWindows()
    return 0
###############################################################################
def write_roi(source, data, cap):
    print("Writing Extracted ROI...")
    starting_directory = os.getcwd() ## Gets the current working directory
    roi_output_folder = 'extracted_roi'
    if (source == "webcam"):
        roi_output_source = source + "_" + roi_output_folder
        roi_output_directory = os.path.join(working_directory, roi_output_source)      ## Defines the name of the directory to be used for storing of ROI frames
    else:
        source_path = source.replace(".mp4", "_").replace(".MP4", "_").replace("\n","")
        roi_output_directory = source_path + roi_output_folder    
    
    if os.path.exists(roi_output_directory):  ## Checks if output directory exists
            shutil.rmtree(roi_output_directory)   ## Deletes directory and all contents
    os.mkdir(roi_output_directory)            ## Creates empty directory
    
    ## Need to write capture source information to a text file
    os.chdir(roi_output_directory)
    cap_output = open("capture_source_info.txt","w+") 
    output_text = [str(cap.source), "\n", str(cap.frames), "\n", str(cap.framerate), "\n"] 
    cap_output.writelines(output_text)
    cap_output.close()
    
    for i in range(len(data)):
        subject_folder = os.path.join(roi_output_directory, str(i)) ## Defines a new folder for each subject
        os.mkdir(subject_folder) ## Creates empty directory
        os.chdir(subject_folder) ## Moves to newly created directory
        for j in range(len(data[i])): ## For every face in list
            filename = str(data[i][j].current_frame) + '.jpg'
            cv.imwrite(filename, data[i][j].roi) ## Write file
    
    os.chdir(starting_directory) ## Returns to working directory at start
    return 0

###############################################################################
def read_roi(source):  
    starting_directory = os.getcwd() ## Gets the current working directory
    roi_input_folder = 'extracted_roi'
    if (source == "webcam"):
        roi_input_source = source + "_" + roi_input_folder
        roi_input_directory = os.path.join(working_directory, roi_input_source)      ## Defines the name of the directory to be used for storing of ROI frames
    else:
        source_path = source.replace(".mp4", "_").replace(".MP4", "_").replace("\n","")
        roi_input_directory = source_path + roi_input_folder    
    
    if os.path.exists(roi_input_directory):  ## Checks if output directory exists
        os.chdir(roi_input_directory)
        
        #print("\nReading ROI from folder...\n")
        
        ## Need to read in capture source information from text file
        cap_input = open("capture_source_info.txt","r+") 
        input_text = cap_input.readlines()
        cap_out = cls.capture_source_data(input_text[0], int(input_text[1]), float(input_text[2]))
        cap_input.close()
        
        directories = [x[1] for x in os.walk(roi_input_directory)] ## Gets all subdirectories
        directories = directories[0]

        output = []
        for i in range(len(directories)):
            subject_folder = os.path.join(roi_input_directory, directories[i]) ## Defines a new folder for each subject
            os.chdir(subject_folder) ## Moves to correct folder
            
            subject_list = []
            file_list = []
            for root, dirs, files in os.walk(subject_folder):
                for file in files:
                    if (file.lower().endswith(".jpg")) | (file.lower().endswith(".jpeg")):
                        file_list.append(file)
            file_list = sorted(file_list,key=lambda x: int(os.path.splitext(x)[0])) ## Sorts the filenames into numerical order
            for j in range(len(file_list)):
                img = cv.imread(file_list[j])
                frame = int(file_list[j].lower().replace(".jpg", "").replace(".jpeg", ""))
                subject_list.append(cls.facial_data(frame, img))
            output.append(subject_list)        
            
    
        os.chdir(starting_directory) ## Returns to working directory at start
        return 1, output, cap_out ## Returns a '1', the list of roi's and capture source info if successful
    else:
        return 0, 0, 0 ## Returns 0 if could not find roi directory

###############################################################################
def resize_by_height(img, height = 200):
    rotated = imutils.rotate_bound(img, 90) ## Rotates 90 degrees
    resized = imutils.resize(rotated, width = height) ## Resizes image using imutils resize function
    output = imutils.rotate_bound(resized, 270) ## Rotates back to original orientation
    return output

###############################################################################
def dataset_source_selection(root_directory, local_directory):
    print("")
    
    video_files = []

    for root, dirs, files in os.walk(local_directory):
        if "raw_video" not in root:
            for file in files:
                if (file.endswith(".mp4")) | (file.endswith(".MP4")):
                    video_files.append([root, file])
    
    for i in range(len(video_files)):
        print(str(i) + ":   Dataset" + video_files[i][0].replace(root_directory, "") + "\\" + video_files[i][1])
    
    
    file_valid = False
    while(file_valid != True):
        file_selection = input("Please select a video file (0-" + str(len(video_files)-1) + "):   ")
                            
        if file_selection == "":
            print("Invalid Argument.")
            file_valid = False
        elif (file_selection.lower() == "q"):
            return 0, 0
        elif (file_selection.isdigit()) | (file_selection == "0"):
            if (int(file_selection) >= 0) & (int(file_selection) < len(video_files)):
                new_source = os.path.join(video_files[int(file_selection)][0], video_files[int(file_selection)][1])
                file_valid = True
            else:
                print("Invalid Argument.")
                file_valid = False
        else:
            print("Invalid Argument.")
            file_valid = False
                        
    return 1, new_source
###############################################################################
def dataset_folder_selection(root_directory):
    for root, dirs, files in os.walk(root_directory):
        print("")
        for i in range(len(dirs)):
            print(i, ":  ", dirs[i])
            
        folder_valid = False
        while(folder_valid != True):
            folder_selection = input("Please select a folder (0-" + str(len(dirs)-1) + "):   ")
                            
            if folder_selection == "":
                print("Invalid Argument.")
                folder_valid = False
            elif (folder_selection.lower() == "q"):
                return 0, 0
            elif (folder_selection.isdigit()) | (folder_selection == "0"):
                if (int(folder_selection) >= 0) & (int(folder_selection) < len(dirs)):
                    new_root = os.path.join(root, dirs[int(folder_selection)])
                    folder_valid = True
                else:
                    print("Invalid Argument.")
                    folder_valid = False            
            else:
                print("Invalid Argument.")
                folder_valid = False
                        
        return 1, new_root
###############################################################################
def build_list_of_faces(source,
                        display_faces = True,
                        cascade_source = default_cascade_xml
                        ):
    
    if (display_faces):    
        selection_valid = False
        while(selection_valid != True):
            user_selection = input("Would you like to write generated list of faces? (y/n):  ")
            if user_selection.lower() == "y":
                write_list_of_faces = True
                selection_valid = True
            elif user_selection.lower() == "n":
                write_list_of_faces = False
                selection_valid = True
            else:
                print("Invalid Argument.")
                selection_valid = False
    else:
        write_list_of_faces = False
        
    if (write_list_of_faces):
        faces_output_folder = "faces_identified"
        if (source == "webcam"):
            faces_output_source = source + "_" + faces_output_folder
            faces_output_directory = os.path.join(working_directory, faces_output_source)      ## Defines the name of the directory to be used for storing of ROI frames
        else:
            source_path = source.replace(".mp4", "_").replace(".MP4", "_")
            faces_output_directory = source_path + faces_output_folder      ## Defines the name of the directory to be used for storing of ROI frames

        if os.path.exists(faces_output_directory):  ## Checks if output directory exists
            shutil.rmtree(faces_output_directory)   ## Deletes directory and all contents
    
        os.mkdir(faces_output_directory)            ## Creates empty directory

    
    
    print("\nBuilding List of Faces...")
    if source == 'webcam':
        cap = cv.VideoCapture(0)  ## Sets capture object to default source
        source_framerate = 15.0     ## Assumes framerate is 15 fps, had issues as using 'cap.get(cv.CAP_PROP_FPS)' always returned 0 
    else:
        cap = cv.VideoCapture(source) ## Sets capture object to provided source path
        source_framerate = cap.get(cv.CAP_PROP_FPS) ## Gets framerate of video source
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) ## Gets total frames from video source
        
    face_cascade = cv.CascadeClassifier(cascade_source) ## Initialises the face cascade to be used
    
    frames_passed = 0 ## Inits counter variables
    faces_found = 0
    
    list_of_faces = [] ## Inits list variables
    frame_faces = []
    missing_frames = []
    
    
    if (cap.isOpened()== False):  ## Error check
        print("Error opening video stream or file. Exiting.")
        cap.release()  ## Releases the video capture object
        os._exit(0)        
    print("Video source opened. Reading...")
    
    
    while(cap.isOpened()): ## Capture frame-by-frame
        ret, frame = cap.read() ## Reads frame by frame, returning boolean and image
        if ret == True:
            
            if (source != 'webcam'):
                if total_frames > 50:  ## Stops divide-by-zero errors
                    if ((frames_passed % int(total_frames/50)) == 0): ## Outputs the status 100 times over runtime
                        print('Percent complete: %.2f' % (100 * frames_passed/total_frames),'%  Faces Identified - ', str(len(list_of_faces)), '/', str(frames_passed))
            
            
            img = np.array(frame) ## Copies the frame so any changes made before drawing do not affect input data
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    ## Converts input frame to grayscale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  ## Detects faces within greyscale image
            
            for (x,y,w,h) in faces:  ## for every face detected in the greyscale image
                faces_found += 1
                roi_colour = frame[y:y+h, x:x+w] ## Gets the region-of-interest in the colour frame
                frame_faces.append(cls.facial_frame(frames_passed, roi_colour, x, y, w, h))  ## Adds the facial region, frame number, and locators on the original image
                cv.rectangle(img,(x,y),(x+w,y+h),(255, 255, 255),3) ## Draw a white rectangle around that face on the input frame
            
            if (faces_found == 0): ## If no face has been found in this frame
                missing_frames.append(cls.full_frame(frames_passed, frame)) ## Save the frame for later use
            else:
                list_of_faces.append(frame_faces)  ## Add all faces found to list of faces
                frame_faces = []  ## Resets variable for next loop
            frames_passed += 1 ## Increments frame counter
            faces_found = 0    ## Resets variable for next pass
            
            if (source == 'webcam'):
                cv.imshow('Webcam Video Source',img) ## Display frame with any faces identified
                if cv.waitKey(10) & 0xFF == ord('q'): ## Waits 10ms and allows to break out of video by pressing q
                    break 
        else:
            if (frames_passed == 0):
                print("Error reading video source")
                cap.release()  ## Releases the video capture object
                cv.destroyAllWindows() ## Closes all OpenCV windows
                os._exit(0)
            break
    
    print("Finished reading video source.")    
    print("\nFrames passed: " + str(frames_passed))
    if (source != 'webcam'):
        print("Frames in source file: " + str(total_frames))
    print("Frames missing an identified face in video source: %d \n" % int(frames_passed - len(list_of_faces)))    
        
    if (write_list_of_faces == True):
        os.chdir(faces_output_directory) ## Changes to directory for ROI frame storage
        for i in range(len(list_of_faces)):
            for j in range(len(list_of_faces[i])):    
                image_name = 'frame_' + str(i) + '_face_' + str(j) + '.jpg'                   ## Defines output filename
                cv.imwrite(image_name, list_of_faces[i][j].facial_data.roi) ## Writes the roi to an image
        os.chdir(working_directory)    
    
    if (display_faces == True):
        j = 0 ## Will only display the first face found in each image
        for i in range(len(list_of_faces)):
            roi = list_of_faces[i][j].facial_data.roi
            cv.imshow('Faces Found', roi)                            ## Display frame with any faces identified
            if cv.waitKey(10) & 0xFF == ord('q'):                    ## Waits 10ms and allows to break out of video by pressing q
                break

    capture_source_info = cls.capture_source_data(source, frames_passed, source_framerate)
    
    cap.release()  ## Releases the video capture object
    cv.destroyAllWindows() ## Closes all OpenCV windows
    
    return list_of_faces, missing_frames, capture_source_info

###############################################################################
def extract_roi(source, 
                display_roi = True,
                height_reduction = 0.0,  ## Sets how much of the height and width of the roi should be cropped
                width_reduction = 0.4    ## Calculated as a percentage i.e. 0.2 = 20% reduction - 10% each side
                ):
    
    list_of_faces, unidentified_frames, capture_source_info = build_list_of_faces(source = source, display_faces = False)
    print("\nExtracting ROI from list of faces...")
    max_faces = 0  ## Determines the maximum number of faces identified in a frame
    avg_face_width = 0
    for i in range(len(list_of_faces)):
        avg_face_width += list_of_faces[i][0].facial_data.roi.shape[0]
        if (max_faces < len(list_of_faces[i])):
            max_faces = len(list_of_faces[i])       
    avg_face_width = int(avg_face_width / len(list_of_faces))
    
    faces = []
    for i in range(max_faces+1): ## +1 to deal with erroneous faces, unused list elements will be deleted later
        faces.append(cls.subject_roi())
                
    for i in range(len(list_of_faces)):
        if (i == 0): ##
            for j in range(len(list_of_faces[i])):    
                faces[j].add_frame(list_of_faces[i][j]) ## Adds the first face(s) in frame 0
        else:
            for f in range(max_faces): ## For each list of faces
                distances = []
                for j in range(len(list_of_faces[i])): ## Calculates distances of identified face location to last in face list
                    if (j == (len(list_of_faces)/50)):
                        print('.', end='')
                    distance = np.linalg.norm(np.array([faces[f].last_x, faces[f].last_y]) - np.array([list_of_faces[i][j].x_loc, list_of_faces[i][j].y_loc]))
                    distances.append(distance)                    
                                
                for t in range(len(distances)):
                    max_distance = 2*avg_face_width ## Defines maximum distance from last face in list to be considered next element, determined experimentally
                    if (distances[t] < max_distance) & (distances[t] == min(distances)): ## If smallest distance & less than maximum distance
                        faces[f].add_frame(list_of_faces[i][t]) ## Add frame to list
                        del list_of_faces[i][t]  ## Delete frame from input list so can't be added again
                    elif (faces[f].last_x == -1): ## If list has no last value
                        faces[f].add_frame(list_of_faces[i][t]) ## Add frame to list
                        del list_of_faces[i][t]  ## Delete frame from input list so can't be added again
    
    i = 0
    deletion_complete = False
    while (deletion_complete != True):  ## Deletes any list of faces generated of length <10% input frames
        if (i >= len(faces)): ## If out of index range
            deletion_complete = True ## Allows exit of loop
        elif (len(faces[i].roi_data) < (0.1 * capture_source_info.frames)): ## Checks if length is greater than 10% the total length of source
            del(faces[i]) ## Deletes the list
            i -= 1  ## By deleting the element, we have made the list one element shorter
        i += 1 ## Increments to next element
    
      
    sorted_faces = []
    for i in range(len(faces)):
        sorted_faces.append(cls.subject_roi())
    
    for i in range(len(faces)): ## for each list of faces
        frame_counter = faces[i].roi_data[0].facial_data.current_frame
        completed = False
        j = 0
        k = 0
        while (completed != True):
            if (frame_counter > (faces[i].roi_data[len(faces[i].roi_data)-1].facial_data.current_frame)):
                completed = True
            elif (faces[i].roi_data[j].facial_data.current_frame == frame_counter): ## if the frame values match
                sorted_faces[i].add_frame(new_frame = faces[i].roi_data[j]) ## Adds the frame to the new list
                j += 1
            else:
                k -= 1
                found = False
                while (found != True): ## Search through list of unidentified frames
                    k += 1
                    if (k >= len(unidentified_frames)):
                        found = True ## Stops errors if frame not in list
                    elif (unidentified_frames[k].current_frame == frame_counter): ## if the frames match
                        x = sorted_faces[i].last_x
                        y = sorted_faces[i].last_y
                        w = sorted_faces[i].roi_data[j-1].width
                        h = sorted_faces[i].roi_data[j-1].height
                        temp_roi = unidentified_frames[k].frame_data[y:y+h, x:x+w]
                        sorted_faces[i].add_frame(new_frame = cls.facial_frame(frame_counter, temp_roi, x, y, w, h)) ## Adds the frame to the new list 
                        found = True
                    
            frame_counter += 1   
  
    ## Resizes all ROI frames to be the same size
    for i in range(len(sorted_faces)):
        avg_w = 0
        for j in range(len(sorted_faces[i].roi_data)):
            avg_w += sorted_faces[i].roi_data[j].width
        avg_w = int(avg_w / len(sorted_faces[i].roi_data))
        
        for j in range(len(sorted_faces[i].roi_data)):
            resized_roi = imutils.resize(sorted_faces[i].roi_data[j].facial_data.roi, width=avg_w) ## Resizes to average size
            
            y = int((height_reduction/2) * resized_roi.shape[0])   ## Sets the variables used to crop the resized roi frame
            h = int((1 - height_reduction) * resized_roi.shape[0])
            x = int((width_reduction/2) * resized_roi.shape[1])
            w = int((1 - width_reduction) * resized_roi.shape[1])
            
            sorted_faces[i].roi_data[j].facial_data.roi = resized_roi[y:y+h, x:x+w] ## Crops and replaces original in list

    output = []
    for i in range(len(sorted_faces)): ## Used to clean up data prior to returning as dont need x,y,h,w, etc
        temp = []
        for j in range(len(sorted_faces[i].roi_data)):
            temp.append(sorted_faces[i].roi_data[j].facial_data)
        output.append(temp)

    if (display_roi == True):             
        display_roi_video(output, capture_source_info)
    if (source != 'webcam'): ## If a video file source
        write_roi(source, output, capture_source_info)
    return output, capture_source_info
###############################################################################
def signal_extract_0(roi_list):
    #qprint("Extracting signals from ROI")
    signals = []
    for i in range(len(roi_list)): ## For every subject
        r_signal = []
        g_signal = []
        b_signal = []
        for j in range(len(roi_list[i])): ## For every face
            r_signal.append(np.average(roi_list[i][j].roi[:,:,2])) ## Averages the red portion of the frame and adds to signal
            g_signal.append(np.average(roi_list[i][j].roi[:,:,1])) ## Averages the green portion of the frame and adds to signal
            b_signal.append(np.average(roi_list[i][j].roi[:,:,0])) ## Averages the blue portion of the frame and adds to signal
        signals.append([r_signal, g_signal, b_signal]) ## Adds the generated signals for this subject to the output list
    
    return signals
###############################################################################
def signal_extract_1(roi_list, rows = 4, cols = 1):
    print("Extracting signals from ROI")
    if rows < 1:
        rows = 1
    if cols < 1:
        cols = 1
    
    output_signals = signal_extract_0(roi_list)
    for i in range(len(roi_list)): ## For every subject
        subject_signals = [[] for x in range(3 * rows * cols)]
        
        for j in range(len(roi_list[i])): ## For every face
            for k in range(0,3): ## For each colour channel, B,G,R in turn
                img_shape = roi_list[i][j].roi[:,:,k].shape
                for r in range(rows):
                    for c in range(cols):
                        subject_signals[(c + (r * cols) + ((k) * rows * cols))].append(np.average(roi_list[i][j].roi[int((r)*(img_shape[0]/rows)):int((r+1)*(img_shape[0]/rows)),int((c)*(img_shape[1]/cols)):int((c+1)*(img_shape[1]/cols)),k]))
        for x in range(len(subject_signals)):
            output_signals[i].append(subject_signals[x])
    return output_signals

###############################################################################
def detrend_signals(signals):
    detrended_signals = []
    for i in range(len(signals)): ## For every subject
        subject_signals = []
        for j in range(len(signals[i])): ## For every signal for the subject
            X = [k for k in range(0, len(signals[i][j]))]
            X = np.reshape(X, (len(X), 1)) ## Reshapes the array into necessary format
            model = LinearRegression() ## Defines the model
            model.fit(X, signals[i][j]) ## Fits signals to the model
            trend = model.predict(X) ## Predicts the model's trend
            detrend = [signals[i][j][k] - trend[k] for k in range(len(signals[i][j]))] ## Removes trend from signal
            subject_signals.append(detrend)
        detrended_signals.append(subject_signals)
    
    
    return detrended_signals

###############################################################################
def normalise_signals(signals):
    normalised_signals = []
    for i in range(len(signals)): ## For every subject
        subject_signals = []
        for j in range(len(signals[i])): ## For every signal for the subject
            signal_mean = np.average(signals[i][j])
            signal_std_dev = np.std(signals[i][j])
            normalised = []
            for k in range(len(signals[i][j])):
                normalised.append((signals[i][j][k] - signal_mean)/signal_std_dev)          
            subject_signals.append(normalised)
        normalised_signals.append(subject_signals)
    
    
    return normalised_signals  
    
###############################################################################
def perform_ica(signals,
                max_iterations = 100000 ## If doesn't converge after 1,000,000 iterations, will likely not converge at all, takes a very long time
                ):
    print("Determining Independent Components (%d signals)" % len(signals[0]))
    ica_signals = []
    for i in range(len(signals)):
        ica = FastICA(n_components=len(signals[i]), max_iter = max_iterations)
        ica_input = [[signals[i][j][k] for j in range(len(signals[i]))] for k in range(len(signals[i][0]))] ## Arranges data as a list of tuples corresponding to the signal value for each frame
        ica_signals.append(ica.fit_transform(ica_input))  # Reconstruct signals
        
    output = []
    for i in range(len(signals)):
        subject_signals = []
        for j in range(len(signals[i])):
            component_signal = []
            for k in range(len(signals[i][j])):
                component_signal.append(ica_signals[i][k,j])
            subject_signals.append(component_signal)
        output.append(subject_signals)
        
    return output

###############################################################################
def bandpass_filter(data, fs=1800, lowcut=10, highcut=100, order=4):
    #print("Filtering signals")
    nyquist = 0.5*fs    ## Calculcates nyquist frequency of the data
    low = lowcut/nyquist    ## Calculates the lower critical frequency
    high = highcut/nyquist  ## Calculates the upper critical frequency     
    b,a = butter(order, [low,high], btype='band')   ## Determines the numerator and denominator polynomials of the filter 
    
    filtered_data = []
    for i in range(len(data)):
        subject_data = []
        for j in range(len(data[i])):
            subject_data.append(lfilter(b, a, data[i][j])) ## Applies the butterworth filter to the data
        filtered_data.append(subject_data)
    return filtered_data

###############################################################################
def moving_average(data, average_length = 5):
    if (average_length < 3):
        average_length = 3
    #print("Performing a %d-point moving average on signals" % average_length)
    averaged_data = []
    for i in range(len(data)):
        subject_data = []
        for j in range(len(data[i])):
            signal_data = []
            for k in range(len(data[i][j])):
                if (k < (average_length/2)):  ## For starting elements
                    average_range = range(0, average_length)
                elif (k + (average_length/2) > len(data[i][j])):
                    average_range = range(-average_length, 0)  ## For end elements
                else:  ## Standard use within list
                    average_range = range(-int(average_length/2), int(average_length/2), +1)
        
                sum_list = [data[i][j][k+l] for l in average_range]
                sum_point = np.sum(sum_list)
                average_point = sum_point / average_length
        
                signal_data.append(average_point)
            subject_data.append(signal_data)
        averaged_data.append(subject_data)
    
    return averaged_data

###############################################################################
def determine_rppg(data):
    print("Determining rPPG signal(s)")
    rppg_signals = []
    for i in range(len(data)):
        max_components = []
        for j in range(len(data[i])):
            fourier_series = abs(np.fft.fft(data[i][j]))
            average = np.average(fourier_series)
            max_components.append([(max(fourier_series)/average), j])
        best_rppg = max(max_components)
        print("Subject %d: " % i, "signal %d chosen..." % best_rppg[1])
        rppg_signals.append(data[i][best_rppg[1]])    
    
    
    return rppg_signals
###############################################################################
def extract_rppg(source, ## Video source path
                 extraction_method = [5, 1], ## Selects which signal extraction method to use
                 plot_signals = False,
                 plot_detrended = False,
                 plot_normalised = False,
                 plot_components = False,
                 plot_filtered = False,
                 plot_averaged = True,
                 plot_rppg = True,
                 ): 
    if (source != 'webcam'): ## If a video file source
        output = read_roi(source) ## Check if the roi has already been extracted for this file
        if output[0] == 1: ## If file has been extracted previously
            roi_list = output[1]
            capture_source_info = output[2]
        else:
            roi_list, capture_source_info = extract_roi(source, display_roi = False)
    else: ## Webcams should always use the extract roi function to extract signal live
        roi_list, capture_source_info = extract_roi(source, display_roi = False)
    
    ## From here need to extract signal from roi, apply initial filtering and then apply ICA
    if (len(extraction_method) == 2):
        signals = signal_extract_1(roi_list, extraction_method[0], extraction_method[1])
    else:
        signals = signal_extract_1(roi_list)
    
    print("Detrending signals")
    detrended_signals = detrend_signals(signals) ## Detrends the provided signals
    print("Normalising signals")
    normalised_signals = normalise_signals(detrended_signals) ## Normalises the detrended signals
    independent_components = perform_ica(normalised_signals) ## Applies Independent Component Analysis
    
    filtered_components = bandpass_filter(data = independent_components, fs = capture_source_info.framerate, lowcut=0.7, highcut=2.4) ## Bandpass filters the data
    averaged_components = moving_average(data = filtered_components, average_length = 5) ## Performs a 5-point moving average
    rppg_signals = determine_rppg(averaged_components) 
    
    ## All after this in function is plotting the data
    try:
        
        if ((plot_signals) | (plot_detrended) | (plot_normalised) | (plot_components) | (plot_filtered) | (plot_averaged)): ## If plotting a signal
            print("\nPlotting Data")
            plots_output_folder = 'plots'
            if (source == "webcam"):
                plots_output_source = source + "_" + plots_output_folder
                plots_output_directory = os.path.join(working_directory, plots_output_source)      ## Defines the name of the directory to be used for storing of ROI frames
            else:
                source_path = source.replace(".mp4", "_").replace(".MP4", "_").replace("\n","")
                plots_output_directory = source_path + plots_output_folder    
        
            if (os.path.exists(plots_output_directory) != True):  ## Checks if output directory exists 
                os.mkdir(plots_output_directory)            ## Creates empty directory if doesn't exist
                
            if (plot_signals): ## If plotting initial signals
                for i in range(len(signals)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                        
                    plot_data = []
                
                    for j in range(len(signals[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(signals[i][j]))]
                        plot_data.append([x, signals[i][j], 'k'])
                    plot_name = "0_initial_signals"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)
            
            if (plot_detrended):
                for i in range(len(detrended_signals)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    for j in range(len(detrended_signals[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(detrended_signals[i][j]))]
                        plot_data.append([x, detrended_signals[i][j], 'k'])
                    plot_name = "1_detrended_signals"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)
                
            if (plot_normalised):
                for i in range(len(normalised_signals)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    for j in range(len(normalised_signals[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(normalised_signals[i][j]))]
                        plot_data.append([x, normalised_signals[i][j], 'k'])
                    plot_name = "2_normalised_signals"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)  
                
            if (plot_components):
                for i in range(len(independent_components)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    for j in range(len(independent_components[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(independent_components[i][j]))]
                        plot_data.append([x, independent_components[i][j], 'k'])
                    plot_name = "3_independent_components"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)
                
            if (plot_filtered):
                for i in range(len(filtered_components)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    for j in range(len(filtered_components[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(filtered_components[i][j]))]
                        plot_data.append([x, filtered_components[i][j], 'k'])
                    plot_name = "4_filtered_signals"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)
                
            if (plot_averaged):
                for i in range(len(averaged_components)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    for j in range(len(averaged_components[i])): ## For every signal in list
                        x = [(k/capture_source_info.framerate) for k in range(0, len(averaged_components[i][j]))]
                        plot_data.append([x, averaged_components[i][j], 'k'])
                    plot_name = "5_averaged_signals"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)                                    
                   
            if (plot_rppg):
                for i in range(len(rppg_signals)):
                    subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
                    if os.path.exists(subject_folder):
                        os.chdir(subject_folder) ## Moves to directory
                    else:
                        os.mkdir(subject_folder) ## Creates empty directory
                        os.chdir(subject_folder) ## Moves to newly created directory
                    
                    plot_data = []
                    
                    x = [(k/capture_source_info.framerate) for k in range(0, len(rppg_signals[i]))]
                    plot_data.append([x, rppg_signals[i], 'k'])
                    
                    fft = abs(np.fft.fft(rppg_signals[i])) ## Calculates the fft of the rPPG
                    freq = [((k * capture_source_info.framerate)/len(rppg_signals[i])) for k in range(0, int(len(fft)/2))] ## Calulates frequencies
                    plot_data.append([freq, fft[0:int(len(fft)/2)], 'k'])
                    
                    plot_name = "6_rPPG_signal"
                    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data)
                os.chdir(working_directory)                
    except:
        print("Error plotting signals")

    
#    for i in range(len(roi_list)): ## Displays each subject identified to the user
#        print("\n\nSubject %d: \n" % i,
#              "First Face Found: %d\n" % roi_list[i][0].current_frame,
#              "Last Face Found: %d\n" % roi_list[i][len(roi_list[i])-1].current_frame,
#              "(%d /" % len(roi_list[i]), ## Number of faces in list
#              "%d)" % ((roi_list[i][len(roi_list[i])-1].current_frame + 1) - roi_list[i][0].current_frame) ## Expected number of faces between last and first in list
#              )
#        name = "Subject " + str(i)
#        cv.imshow(name, imutils.resize(roi_list[i][0].roi, width = 300))
#    if cv.waitKey(10000) & 0xFF == ord('q'):                    ## Waits 10s and allows to break out of video by pressing q
#        cv.destroyAllWindows()
#    cv.destroyAllWindows()    
    
    
    return rppg_signals, capture_source_info

###############################################################################
def get_PPG(ppg_file = None ## Expects list of two strings: [[directory of PPG file] [PPG filename]]
):
    if (ppg_file == None):
        print("Must provide a valid PPG file")

    red_signal = []
    infra_signal = []
	
    with open(os.path.join(ppg_file[0][0], ppg_file[0][1])) as csvfile:
        read_csv = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in read_csv:
            i += 1
            if i >= 7:
                red_signal.append(float(row[4]))
                infra_signal.append(float(row[5]))
	
    return red_signal, infra_signal

###############################################################################
def compare_ppg(rppg_signals, cap):
    
    ppg_file = []
    temp_path = os.path.abspath(os.path.join(os.path.dirname(cap.source), '..', '..', '..'))
    ppg_path = os.path.join(temp_path, 'ppg_data')
    
    for root, dirs, files in os.walk(ppg_path):
        if "raw_video" not in root:
            for file in files:
                if (file.endswith(".xls")):
                    if ('Volts' in file):
                        ppg_file.append([root, file])
    
    if len(ppg_file) > 1:
        print("Error: Multiple PPG recordings detected")
        return 0
    elif len(ppg_file) != 1:
        print("Error: PPG recordings could not be detected")
        return 0
    
    red_signal, infra_signal = get_PPG(ppg_file) 
    fft_red = abs(np.fft.fft(red_signal))
    fft_infra = abs(np.fft.fft(infra_signal))
    
     
    if ((max(fft_red) <  max(fft_infra))):
        ppg_signal = red_signal  
    else:
        ppg_signal = infra_signal
    resampled_ppg = [] 
    temp = []
    temp.append(list(sps.resample(ppg_signal, int(cap.framerate*(len(ppg_signal)/500)))))
    resampled_ppg.append(temp)
    detrended_ppg = detrend_signals(resampled_ppg)
    normalised_ppg = normalise_signals(detrended_ppg)
    filtered_ppg = bandpass_filter(data = normalised_ppg, fs = cap.framerate, lowcut=0.7, highcut=2.4) ## Bandpass filters the data
    final_ppg = filtered_ppg[0][0]
    
    final_ppg = final_ppg[int(5*cap.framerate):int((5*cap.framerate) + len(rppg_signals[0]))] ## Sets the final PPG to be the same length of the rPPG signal, starting 5 seconds after the start of the PPG
    
    plots_output_folder = 'plots'
    if (cap.source == "webcam"):
        plots_output_source = cap.source + "_" + plots_output_folder
        plots_output_directory = os.path.join(working_directory, plots_output_source)      ## Defines the name of the directory to be used for storing of ROI frames
    else:
        source_path = cap.source.replace(".mp4", "_").replace(".MP4", "_").replace("\n","")
        plots_output_directory = source_path + plots_output_folder    
           
    subject_folder = os.path.join(plots_output_directory, '0') ## Assume only one subject in video if comparing ppg and rppg signal
    os.chdir(subject_folder) ## Moves to newly created directory
                    
    plot_data = []
    plot_name = "rPPG_to_PPG_comparison"    
        
    x = [(k/cap.framerate) for k in range(0, len(rppg_signals[0]))]
    plot_data.append([x, rppg_signals[0], 'k'])
    
    y = [(k/cap.framerate) for k in range(0, len(final_ppg))]
    plot_data.append([y, final_ppg, 'k'])
    
    save_graph_plot(subject_folder, plot_name, len(plot_data), plot_data, silent = True)
           
    os.chdir(working_directory)
    
    return [red_signal, infra_signal], final_ppg

###############################################################################
def determine_hr_fft(signal, cap):
    fft = abs(np.fft.fft(signal)) ## Calculates the fft
    fft = fft[0:int(len(fft/2))] ## Eliminates all reflections
    freq = [((k * cap.framerate)/len(signal)) for k in range(0, len(fft))] ## Calulates frequencies
    fft_peak = np.where(fft == np.amax(fft))[0][0] ## Finds the fourier peak
    hr_freq = freq[fft_peak] ## Determines the corresponding frequency
    return (hr_freq * 60)

###############################################################################
def determine_hr_peak(signal, cap):
    maxima, minima  = peak.peakdet(signal, (max(signal)/8))
    indices = list(maxima[:,0]) ## Gets the indexes of the maximum points
    indices = [int(indices[j]) for j in range(len(indices))] ## Typecasts list as integers
               
    heart_rate = 60 * (len(indices)/((indices[len(indices)-1] - indices[0])/cap.framerate))
    return heart_rate

###############################################################################
def determine_heart_rate(rppg_signals, capture_source_info):
    
    plots_output_folder = "plots"
    if (capture_source_info.source == "webcam"):
        plots_output_source = capture_source_info.source + "_" + plots_output_folder
        plots_output_directory = os.path.join(working_directory, plots_output_source)      ## Defines the name of the directory to be used for storing of ROI frames
    else:
        source_path = capture_source_info.source.replace(".mp4", "_").replace(".MP4", "_").replace("\n","")
        plots_output_directory = source_path + plots_output_folder
    
   
    
    for i in range(len(rppg_signals)):
        subject_folder = os.path.join(plots_output_directory, str(i)) ## Defines a new folder for each subject
        if os.path.exists(subject_folder):
            os.chdir(subject_folder) ## Moves to directory
        else:
            os.mkdir(subject_folder) ## Creates empty directory
            os.chdir(subject_folder) ## Moves to newly created directory
        
        print("\nSubject %d:\n" % i)
        
        rppg_heart_rates = []
        hr_fft = determine_hr_fft(rppg_signals[i], capture_source_info)
        hr_peaks = determine_hr_peak(rppg_signals[i], capture_source_info)
        
        rppg_heart_rates.append(hr_fft)
        rppg_heart_rates.append(hr_peaks)
        
        print("rPPG FFT HR estimation:   %.2f" % hr_fft)
        print("rPPG Peaks HR estimation: %.2f" % hr_peaks)
        
        hr_output = open("hr_info.txt","w+")
        hr_output.writelines(['rPPG HRs:  ', str(hr_fft), '    ', str(hr_peaks), '\n'])
        ppg_heart_rates = []
        ppg_out = compare_ppg(rppg_signals, capture_source_info)
        if (ppg_out != 0):
            resampled_ppg = ppg_out[1]
            ppg_hr_fft = determine_hr_fft(resampled_ppg, capture_source_info)
            ppg_hr_peaks = determine_hr_peak(resampled_ppg, capture_source_info)
            ppg_heart_rates.append(ppg_hr_fft)
            ppg_heart_rates.append(ppg_hr_peaks)
        
            print("\nPPG HR FFT calculated:   %.2f" % ppg_hr_fft)
            print("PPG HR Peak calculated:  %.2f" % ppg_hr_peaks)
            hr_output.writelines(['PPG HRs:   ', str(ppg_hr_fft), '    ', str(ppg_hr_peaks), '\n'])
        ## Need to output HR info to text file
         
      
        hr_output.close()
    
    
    os.chdir(working_directory)
    
    
    return rppg_heart_rates, ppg_heart_rates
