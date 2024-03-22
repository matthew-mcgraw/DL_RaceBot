"""
2023 Spring CSCI 4302/5302 Autonomous Vehicle Challenge
University of Colorado Boulder

Sample Controller
(C) 2023 Bradley Hayes <bradley.hayes@colorado.edu>
"""

from __future__ import annotations
from typing import Callable
from controller import Robot
import math

# create the Robot instance.
robot = Robot()

# Simulation Timestep -- Gets the time step for the current world (Set in WorldInfo).
timestep = int(robot.getBasicTimeStep())


'''
Initialize Sensors and Motors
'''
camera = robot.getDevice('camera')
rear_camera = robot.getDevice('rear_camera')
# Not utilized: rear_camera
lidar = robot.getDevice('LDS-01')
print(lidar.getMinRange())
print(lidar.getMaxRange())

steer_left_motor = robot.getDevice('left_steer')
steer_right_motor = robot.getDevice('right_steer')
l_motor = robot.getDevice('rwd_motor_left')
r_motor = robot.getDevice('rwd_motor_right')

for motor in [l_motor, r_motor]:
    motor.setPosition(float('inf'))
    motor.setVelocity(0)

camera.enable(timestep)
rear_camera.enable(timestep)
lidar.enable(timestep) # 100ms LIDAR Readings
lidar.enablePointCloud()


'''
Convenience functions for your controller
'''
def set_velocity(v: float):
    '''
    Sets rotational velocity of the rear wheel drive motors to v radians/second.
    '''
    for motor in [l_motor, r_motor]:
        motor.setVelocity(v)
     
def set_steering_angle(angle_rad: float):
    '''
    Sets front wheel directions to appropriate angles given their horizontal wheel distances
    for an Ackermann vehicle.
    '''
    trackFront = 0.254
    wheelbase = 0.2921
    angle_right = 0
    angle_left = 0
    if math.fabs(angle_rad) > 1e-5:   
        angle_right = math.atan(1. / (1./math.tan(angle_rad) - trackFront / (2 * wheelbase)));
        angle_left = math.atan(1. / (1./math.tan(angle_rad) + trackFront / (2 * wheelbase)));
    steer_right_motor.setPosition(angle_right)
    steer_left_motor.setPosition(angle_left)


# Main loop:
# Calls to robot.step advance the simulation one step
while robot.step(timestep) != -1:
    set_steering_angle(math.radians(20))
    set_velocity(60)
    lidar_rays = lidar.getRangeImage()
    camera_image = rear_camera.getFov()
    #print(camera_image)
