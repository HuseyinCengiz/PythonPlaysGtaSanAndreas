import numpy as np
import cv2
import mss
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
from statistics import mean
from numpy import ones,vstack
from numpy.linalg import lstsq

def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:              
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                
                m, b = lstsq(A, y_coords)[0]
                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        
        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
        
        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]
        
        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        print(str(e))


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

lower_yellow = np.array([7,73,55])  # BGR-code of your lowest yellow
upper_yellow = np.array([27,93,135])   # BGR-code of your highest yellow
vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])
increase=30

def process_img(original_image):
    original_img_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
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
    lines = cv2.HoughLinesP(added_images,1,np.pi/180,180,None,200,5)
    try:
        l1, l2 = draw_lanes(original_image,lines)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)  
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return processed_img,original_image


def main():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 580}
        
        while(True):
            last_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img_screen = np.array(sct.grab(monitor))
            new_screen,original_image = process_img(img_screen)
            
            #cv2.imshow('window', new_screen)
            cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            
            print("fps: {}".format(1 / (time.time() - last_time)))
            
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

main()
