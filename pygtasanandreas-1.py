import numpy as np
import cv2
import mss
import time

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
    
    while(True):
        last_time = time.time()

         # Get raw pixels from the screen, save it to a Numpy array
        img_screen = np.array(sct.grab(monitor))
    
        # Display the picture
        cv2.imshow('window',img_screen)
        
        print("fps: {}".format(1 / (time.time() - last_time)))
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
