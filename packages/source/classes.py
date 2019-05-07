# Some kind of license?
#
#
#
#
###############################################################################

class capture_source_data(object):
    ## Initialisation script
    def __init__(self, source_path, frames, fps):
        self.source = source_path
        self.frames = frames
        self.framerate = fps
        
class full_frame(object):
    def __init__(self, frame, data):
        self.current_frame = frame
        self.frame_data = data

class facial_data(object):
    ## Initialisation script
    def __init__(self, frame, data):
        self.current_frame = frame
        self.roi = data
        
class facial_frame(object):
    ## Initialisation script
    def __init__(self, frame, data, x, y, w, h):
        self.facial_data = facial_data(frame, data)
        self.x_loc = x
        self.y_loc = y
        self.width = w
        self.height = h

class subject_roi(object):
    ## Initialisation script
    def __init__(self):
        self.roi_data = []
        self.last_x = -1
        self.last_y = -1
        
    def add_frame(self, new_frame):
        self.roi_data.append(new_frame)
        current_pos = int(len(self.roi_data) - 1)
        self.last_x = self.roi_data[current_pos].x_loc
        self.last_y = self.roi_data[current_pos].y_loc
        