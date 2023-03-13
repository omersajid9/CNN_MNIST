
# from import_list import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# import utils.Data


from utils.Data import Data as Data
from utils.layers import Conv, Maxpool, ReLU, tanh, FC, softmax, sigmoid, Dropout, Flatten, BatchNorm
from utils.Network import LeNet5
from flask import Flask, jsonify, render_template, request
import cv2 
# from Herminte import Hermite_class
# import Hermite_class from 'Hermite'
