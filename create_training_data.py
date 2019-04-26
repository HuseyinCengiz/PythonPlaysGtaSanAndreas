import numpy as np
import cv2
import mss
import time
from getkeys import key_check
from os import listdir
from os.path import isfile, join
import re

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]



def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0  1  2  3  4   5   6   7    8
    [W, S, A, D, WA, WD, SA, SD, NOKEY] boolean values.
    '''
    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
    return output

        


def main():

    training_data = []
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 600}
        
        while(True):

            if not paused:
            
                last_time = time.time()
                # Get raw pixels from the screen, save it to a Numpy array
                screen = np.array(sct.grab(monitor))
                screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
                screen = cv2.resize(screen,(160,120))
                keys = key_check()
                output = keys_to_output(keys)
                training_data.append([screen,output])
                print("fps: {}".format(1 / (time.time() - last_time)))

                if len(training_data) % 5000 == 0:
                    print(len(training_data))
                    training_datas = [file for file in listdir('.') if isfile(join('.', file)) if re.match('training_data-[0-9].npy',file)]
                    np.save('training_data-{}.npy'.format(len(training_datas)+1),training_data)
                    training_data = []
                    
            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print('resuming')
                    for i in list(range(3))[::-1]:
                        print(i+1)
                        time.sleep(1)
                else:
                    print("pausing")
                    paused = True
                    time.sleep(1)
                
                             
main()
