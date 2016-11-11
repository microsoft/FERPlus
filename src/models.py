
import os
import sys
import math
import numpy as np

from cntk.blocks import default_options
from cntk.layers import Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense
from cntk.models import Sequential, LayerStack
from cntk.utils import *
from cntk.initializer import glorot_uniform
from cntk import Trainer
from cntk.learner import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk.ops import input_variable, constant, parameter, relu

def build_model(num_classes, model_name):
    '''
    Factory function to instantiate the model.
    '''
    model = getattr(sys.modules[__name__], model_name)
    return model(num_classes)

class VGG13(object):
    '''
    A VGG13 like model (https://arxiv.org/pdf/1409.1556.pdf) tweaked for emotion data.
    '''
    @property
    def learning_rate(self):
        return 0.05

    @property
    def input_width(self):
        return 64

    @property
    def input_height(self):
        return 64

    @property
    def input_channels(self):
        return 1

    @property
    def model(self):
        return None

    @property
    def model(self):
        return self._model

    def __init__(self, num_classes):
        self._model = self._create_model(num_classes)

    def _create_model(self, num_classes):
        with default_options(activation=relu, init=glorot_uniform()):
            model = Sequential([
                LayerStack(2, lambda i: [
                    Convolution((3,3), [64,128][i], pad=True),
                    Convolution((3,3), [64,128][i], pad=True),
                    MaxPooling((2,2), strides=(2,2)),
                    Dropout(0.25)
                ]),
                LayerStack(2, lambda i: [
                    Convolution((3,3), [256,256][i], pad=True),
                    Convolution((3,3), [256,256][i], pad=True),
                    Convolution((3,3), [256,256][i], pad=True),                
                    MaxPooling((2,2), strides=(2,2)),
                    Dropout(0.25)
                ]),            
                LayerStack(2, lambda : [
                    Dense(1024),
                    Dropout(0.5)
                ]),
                Dense(num_classes, activation=None)
            ])

        return model
