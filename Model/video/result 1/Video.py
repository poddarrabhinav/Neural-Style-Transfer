import  numpy as np
import cv2
import os
from utils import CFEVideoConf,image_resize
cap = cv2.VideoCapture(0)
save_path = 'Video/timelapse.mp4'
fps = 24.0
config = CFEVideoConf
