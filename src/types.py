# Zach Blum, Navjot Singh, Aristos Athens

'''
    Types for use in other packages.
    Put these in this separate file to avoid circular dependencies.
'''

import enum


# ------------------------------------- Types ------------------------------------- #
  
class SensorType(enum.Enum): 
    temp = 0
    accel = 1
    gyro = 2
    magnet = 3

class BodyPart(enum.Enum):
    hand = 0
    chest = 1
    ankle = 2