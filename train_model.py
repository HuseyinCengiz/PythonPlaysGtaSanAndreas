#train_model.py
import numpy as np
from models import alexnet2
from os import listdir
from os.path import isfile, join
import re

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs.model'.format(LR,'alextnet2',EPOCHS)

model = alexnet2(WIDTH,HEIGHT,LR,output=9)

training_datas = [file for file in listdir('.') if isfile(join('.', file)) if re.match('training_data-[0-9]-balanced.npy',file)]

numberOfTrainingData = len(training_datas)+1

for i in range(1,EPOCHS):
    data_order = [i for i in range(1,numberOfTrainingData)]
    shuffle(data_order)
    for i in data_order:
        
        train_data = np.load('training_data-{}.npy'.format(i))

        train = train_data[:-100] # son 100 item hariç hepsi
        test = train_data[-100:] #son 100 item

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        '''
        model.fit(X, Y, n_epoch=EPOCHS, validation_set=(test_x,test_y),
                  show_metric=True,snapshot_step=500, run_id=MODEL_NAME)
        '''

        model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                  validation_set=({'input': test_x}, {'targets': test_y}), 
                  snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)




# tensorboard --logdir=foo:C:/Users/HuseyinCengiz/Desktop/PythonPlaysGtaSanAndreas/PythonPlaysGtaSanAndreas/log






