import numpy as np
import cv2
import mss
import time
from getkeys import key_check
from models import alexnet2
from directkeys import PressKey, ReleaseKey, W, A, S, D
import os
 
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR,'alextnet2',EPOCHS)


w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(D)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

model = alexnet2(WIDTH, HEIGHT, LR, output=9)
model.load(MODEL_NAME)

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

        paused = False
        mode_choice = 0
        
        while(True):
            if not paused:
                last_time = time.time()
                # Get raw pixels from the screen, save it to a Numpy array
                screen = np.array(sct.grab(monitor))
                screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
                screen = cv2.resize(screen,(160,120))
                
                print("fps: {}".format(1 / (time.time() - last_time)))

                prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
                print(prediction)
                                
                mode_choice = np.argmax(prediction)
                choice_picked='none'
                
                if mode_choice == 0:
                    straight()
                    choice_picked = 'straight'
                elif mode_choice == 1:
                    reverse()
                    choice_picked = 'reverse'
                elif mode_choice == 2:
                    left()
                    choice_picked = 'left'
                elif mode_choice == 3:
                    right()
                    choice_picked = 'right'
                elif mode_choice == 4:
                    forward_left()
                    choice_picked = 'forward+left'
                elif mode_choice == 5:
                    forward_right()
                    choice_picked = 'forward+right'
                elif mode_choice == 6:
                    reverse_left()
                    choice_picked = 'reverse+left'
                elif mode_choice == 7:
                    reverse_right()
                    choice_picked = 'reverse+right'
                elif mode_choice == 8:
                    no_keys()
                    choice_picked = 'nokeys'
                    
                print("fps: {}. Choice: {}".format(1 / (time.time() - last_time),choice_picked))

            keys = key_check()

            if 'T' in keys:
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    ReleaseKey(A)
                    ReleaseKey(W)
                    ReleaseKey(D)
                    time.sleep(1)
                
            

main()
