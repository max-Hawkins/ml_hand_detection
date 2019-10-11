import argparse
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import cv2
from pynput.keyboard import Key, Listener
from datetime import datetime
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="relative file path to model to use in deployment", type=str)
args = parser.parse_args()

classes = ['Attentive','Distracted']
model = load_model(args.m)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())            
            
        depth_image = cv2.resize(depth_image, dsize=(300,300))
        color_image = cv2.resize(color_image, dsize=(300,300))
        
        depth_image_edit = cv2.convertScaleAbs(depth_image, alpha=0.18)

        model_input = np.stack((depth_image_edit,)*3, axis=-1).reshape(-1,300,300,3)

        prediction = model.predict(model_input)
        
        # Show images
        cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
        # Displays the predicted label on the color image
        cv2.putText(color_image, classes[np.argmax(prediction)],(40,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('Color', color_image)

        cv2.namedWindow('Depth Edit', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Edit', depth_image_edit)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break    

finally:
    # Stop streaming
    pipeline.stop()