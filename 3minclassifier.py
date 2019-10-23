import picamera
import picamera.array
import time
import datetime
import tensorflow as tf
import numpy as np
#import cv2


IMG_SIZE = 50
MODEL_NAME = "8-17-2conv-7epoch-model"
model = tf.keras.models.load_model("./{modelname}".format(modelname = MODEL_NAME))
counter = 0
history = [None] * 10
outfilename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(outfilename, 'w') as f:
    f.write('timestamp,label,#of dry caputres in the past 10\n')
print(outfilename + " is the name of the out file for this run")
with picamera.PiCamera() as camera:
    camera.iso = 1
    camera.brightness = 30
    camera.color_effects = (128,128)
    #camera.start_preview()
    # Camera warm-up time/setup time -- 15 seconds from starting the program to get it somewhere good
    # Copy over camera settings from our training data collection
    time.sleep(2)
    #camera.stop_preview()
    camera.start_recording("{}.h264".format(outfilename))
    while counter < 72:
        with picamera.array.PiRGBArray(camera, size=(IMG_SIZE, IMG_SIZE)) as stream:
            camera.capture(stream, 'rgb', resize=(IMG_SIZE, IMG_SIZE))
            # At this point the image IN 3 CHANNELS is available as stream.array
            image = stream.array
            stream.truncate() #clear the stream for next capture
            stream.seek(0) # go back to start of stream and overwrite it
        image = np.dot(image, [.3, .6, .1]) #convert the 3 channel array to a greyscale 50x50x1
        batch = image.reshape(1, IMG_SIZE, IMG_SIZE, 1) #let tensorflow know there's only one image by padding a 1
        model_out = model.predict(batch)[0]
        if model_out[0] > .5:
            label = "Dry " + str(model_out[0])
            history[counter % 10] = 1
            
        else:
            label = "Wet " + str(model_out[0])
            history[counter % 10] = 0
        print(label)
        drynum = "Not enough data"
        if counter > 8:
            ratio = np.bincount(history, minlength=2)
            print("The ratio of dry to wet classifications in the past 10 captures is " + str(ratio[1]) + " to " + str(ratio[0]))
            drynum = str(ratio[1])
        with open(outfilename, 'a') as f:
            t = datetime.datetime.now().strftime("%H:%M:%S")
            f.write("{},{},{}\n".format(t,label,drynum))
        time.sleep(2.5)
        counter += 1
    camera.stop_recording()    
    
