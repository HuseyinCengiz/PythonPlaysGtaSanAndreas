import numpy as np
import cv2
import mss
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

def draw_lines(img, lines):
    try:
        for line in lines:
            coord = line[0]
            cv2.line(img,(coord[0],coord[1]),(coord[2],coord[3]),[255,255,255],3)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

lower_yellow = np.array([7,73,55])  # BGR-code of your lowest yellow
upper_yellow = np.array([27,93,135])   # BGR-code of your highest yellow
vertices = np.array([[10,500],[10,300],[300,250],[500,250],[800,300],[800,500]])
increase=30

def process_img(original_img):
    original_img_hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    v = original_img_hsv[:, :, 2]
    v = np.where(v <= 255 - increase, v + increase, 255)
    original_img_hsv[:, :, 2] = v
    original_img_bgr = cv2.cvtColor(original_img_hsv, cv2.COLOR_HSV2BGR)
    
    middle_yellow_mask = cv2.inRange(original_img_hsv, lower_yellow, upper_yellow)  
    processed_middle_yellow_mask = roi(middle_yellow_mask, [vertices])
    
    processed_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img,(5,5),5)
    processed_img = roi(processed_img, [vertices])

    added_images = cv2.add(processed_middle_yellow_mask,processed_img)
    # edges
    lines = cv2.HoughLinesP(added_images,1,np.pi/180,180,None,40,25)
    draw_lines(added_images, lines)
    return added_images


def main():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
        
        while(True):
            last_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img_screen = np.array(sct.grab(monitor))
            new_img_screen = process_img(img_screen)
            # Display the picture
            cv2.imshow('window',new_img_screen)
            #cv2.imshow('window2',img_screen)
            
            print("fps: {}".format(1 / (time.time() - last_time)))
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

main()
