"""MovingObstacleController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Supervisor
import numpy as np

# create the Robot instance.
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
bot_node = robot.getFromDef("MovingObstacle")

def get_translation(node):
    trans_field = node.getField("translation")
    values = trans_field.getSFVec3f()
    #print("OBSTACLE is at position: %g %g %g" % (values[0], values[1], values[2]))
    return values[0], values[1], values[2]

def translate_x(node,increment):
    trans_field = node.getField("translation")
    values = trans_field.getSFVec3f()
    new_x = values[0]+increment
    node.getField("translation").setSFVec3f([new_x,values[1],values[2]])
    
def translate_y(node,increment):
    trans_field = node.getField("translation")
    values = trans_field.getSFVec3f()
    new_y = values[1]+increment
    node.getField("translation").setSFVec3f([values[0],new_y,values[2]])

# Main loop:
# - perform simulation steps until Webots is stopping the controller
direction_flag = 0
x_increment = 0.01
y_increment = 0.01

x,y,z = get_translation(bot_node)
rand_x = np.random.uniform(10.9,14.5)
print(rand_x)

bot_node.getField("translation").setSFVec3f([rand_x,y,z])
while robot.step(timestep) != -1:


    x,y,z = get_translation(bot_node)
    
    if y > 11:
        translate_y(bot_node,-21)
    else:
        translate_y(bot_node,y_increment)
            
    
    
    

# Enter here exit cleanup code.
