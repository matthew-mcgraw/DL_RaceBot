from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent_modifed import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
import math
import time
import statistics

#from controller import Robot

#This version will only use the lidar sensor, location, and orientation of the simple vehicle


class Learn2Drive(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        #self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
        #                             high=np.array([0.4, np.inf, 1.3, np.inf]),
        #                             dtype=np.float64)
        
        #For the BB8 robot in this environment, it will be x pos (-49, 49), y pos (-49, 49), MAYBE velocity
        #also let's use the rotation information
        # [xPos, yPos, xAng, yAng, zAng, Angle, lidar_distances]

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        
        self.robot_node = self.getFromDef("SimpleVehicle")
        
        trans_field = self.robot_node.getField("translation")
        values = trans_field.getSFVec3f() 
        print(values)       
        
        self.prev_x = 0
        self.prev_y = 0
        timestep = int(self.getBasicTimeStep())
        
        self.camera = self.getDevice('camera')
        # Not utilized: rear_camera
        self.lidar = self.getDevice('LDS-01')
        print(self.lidar.getMinRange())
        print(self.lidar.getMaxRange())
        
        self.steer_left_motor = self.getDevice('left_steer')
        self.steer_right_motor = self.getDevice('right_steer')
        self.l_motor = self.getDevice('rwd_motor_left')
        self.r_motor = self.getDevice('rwd_motor_right')
        
        for motor in [self.l_motor, self.r_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)
        
        self.camera.enable(timestep)
        self.lidar.enable(timestep) # 100ms LIDAR Readings
        self.lidar.enablePointCloud()    
       
        self.steps_per_episode = 50000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.DRIVE_MAX_SPEED = 60
        self.STEER_MAX_ANGLE = math.radians(35)
        
        lidar_rays = self.lidar.getRangeImage()
        self.num_lidar_rays = len(lidar_rays)
        print(lidar_rays)
        lidar_min = np.ones(len(lidar_rays)) * 0.1
        lidar_max = np.ones(len(lidar_rays)) * 4
        print(lidar_max)
        
        
                
        #past start, past 1, past 2, past 3, past goal again -> one lap completed
        self.check_point_list_min = list(np.zeros(5))
        self.check_point_list_max = list(np.ones(5))
        self.check_point_list = list(np.zeros(5))
        self.completed_lap = False
        print(self.check_point_list)
        
        
        obs_space_min = [-15, -15, -1, -1, -1, -math.pi] + list(lidar_min) + self.check_point_list_min
        obs_space_max = [15, 15, 1, 1, 1, math.pi] + list(lidar_max) + self.check_point_list_max
        print(obs_space_min)
        print(obs_space_max)
        
        self.observation_space = Box(low=np.array(obs_space_min),
                                     high=np.array(obs_space_max),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        #do nothing, drive forward, turn left, turn right
        self.action_space = Discrete(4)
    
    def set_velocity(self, v: float):
        '''
        Sets rotational velocity of the rear wheel drive motors to v radians/second.
        '''
        for motor in [self.l_motor, self.r_motor]:
            motor.setPosition(float('inf'))
            motor.setVelocity(v)
     
    def set_steering_angle(self,angle_rad: float):
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
        self.steer_right_motor.setPosition(angle_right)
        self.steer_left_motor.setPosition(angle_left) 
            
    def get_translation(self):
        trans_field = self.robot_node.getField("translation")
        values = trans_field.getSFVec3f()
        #print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
        return values[0], values[1], values[2]
        
    def get_rotation(self):
        rot_field = self.robot_node.getField("rotation")
        values = rot_field.getSFVec3f()
        #print("MY_ROBOT is at rotation: %g %g %g %g" % (values[0], values[1], values[2], values[3]))
        return values[0], values[1], values[2], values[3]
        
    def intersecting_obstacle_plane(self):
        obs_trans_field = self.obstacle_node.getField("translation")
        obs_trans_values = obs_trans_field.getSFVec3f()
        
        obs_upper_x = obs_trans_values[0] + 13/2 + 0.25
        obs_lower_x = obs_trans_values[0] - 13/2 - 0.25
        obs_upper_y = obs_trans_values[1] + 1/2 + 0.25
        obs_lower_y = obs_trans_values[1] - 1/2 - 0.25
                
        robot_trans_field = self.robot_node.getField("translation")
        robot_trans_values = robot_trans_field.getSFVec3f()
        robot_x = robot_trans_values[0]
        robot_y = robot_trans_values[1]
        
        
        if(robot_x >= obs_lower_x and robot_x <= obs_upper_x and robot_y >= obs_lower_y and robot_y <= obs_upper_y):
            return True
        else:
            return False
    
    #This function checks if the car has past a checkpoint and changes the value from 0 to 1 if it has
    #also adds a score of +1 to self.episode_score
    def check_for_checkpoints(self):
        chkPts = self.check_point_list
        x, y, z = self.get_translation()
        
        if chkPts[0] == 0:
            if y > 0 and self.prev_y > 0 and x > 0 and self.prev_x < 0:
                chkPts[0] = 1.0
                print(chkPts)
                self.episode_score += 1
                print("Past first checkpoint!")
                self.prev_x = x
                self.prev_y = y
                return chkPts
            return chkPts
        elif chkPts[1] == 0:
            if y < 0 and self.prev_y > 0 and x > 0 and self.prev_x > 0:
                chkPts[1] = 1.0
                print(chkPts)
                self.episode_score += 1
                print("Past second checkpoint!")
                return chkPts
            return chkPts
        elif chkPts[2] == 0:
            if y < 0 and self.prev_y < 0 and x < 0 and self.prev_x > 0:
                chkPts[2] = 1.0
                print(chkPts)
                self.episode_score += 1
                print("Past third checkpoint!")
                self.prev_x = x
                self.prev_y = y
                return chkPts
            return chkPts
        elif chkPts[3] == 0:
            if y > 0 and self.prev_y < 0 and x < 0 and self.prev_x < 0:
                chkPts[3] = 1.0
                print(chkPts)
                self.episode_score += 1
                print("Past fourth checkpoint!")
                self.prev_x = x
                self.prev_y = y
                return chkPts
            return chkPts
        elif chkPts[4] == 0:
            if y > 0 and self.prev_y > 0 and x > 0 and self.prev_x < 0:
                chkPts[4] = 1.0
                print(chkPts)
                self.episode_score += 1
                print("Past final checkpoint!")
                self.prev_x = x
                self.prev_y = y
                self.completed_lap = True
                return chkPts
            return chkPts
        else:
            self.prev_x = x
            self.prev_y = y
            return chkPts
            
        
        
    def intersecting_goal_plane(self):
        goal_trans_field = self.goal_node.getField("translation")
        goal_trans_values = goal_trans_field.getSFVec3f()
        
        goal_upper_x = goal_trans_values[0] + 5.5/2 + 0.25
        goal_lower_x = goal_trans_values[0] - 5.5/2 - 0.25
        goal_upper_y = goal_trans_values[1] + 1/2 + 0.25
        goal_lower_y = goal_trans_values[1] - 1/2 - 0.25
                
        robot_trans_field = self.robot_node.getField("translation")
        robot_trans_values = robot_trans_field.getSFVec3f()
        robot_x = robot_trans_values[0]
        robot_y = robot_trans_values[1]
                
        if(robot_x >= goal_lower_x and robot_x <= goal_upper_x and robot_y >= goal_lower_y and robot_y <= goal_upper_y):
            return True
        else:
            return False

    def get_observations(self):
        
        x, y, z = self.get_translation()

        x = normalize_to_range(x, -30, 30, -1.0, 1.0)
        y = normalize_to_range(y, -30, 30, -1.0, 1.0)
        
        lidar_rays = self.lidar.getRangeImage()
        #print(lidar_rays)
        lidar_rays_norm = np.ones(self.num_lidar_rays)
        for i, ray in enumerate(lidar_rays):
            if ray > 4:
                ray = 4
            lidar_rays_norm[i] = normalize_to_range(ray, 0.1, 4, -1, 1)
           
        xAng, yAng, zAng, Angle = self.get_rotation()
        Angle = normalize_to_range(Angle, -math.pi, math.pi, -1.0, 1.0)
        
        chkPts = self.check_for_checkpoints()
        
        #print(self.intersecting_obstacle_plane())
        #print(self.intersecting_goal_plane())
        obs = [x,y,xAng,yAng,zAng,Angle] + list(lidar_rays_norm) + chkPts
        #print(obs)

        return obs

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation

        x, y, z = self.get_translation()
        self.starting_x = x
        self.starting_y = y
        self.prev_x = x
        self.prev_y = y
        self.check_point_list = list(np.zeros(5))
        self.completed_lap = False
        #print(self.starting_x, self.starting_y)
        return [0.0 for _ in range(self.observation_space.shape[0])]
    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        #return 1
        x,y,z = self.get_translation()
        dist = math.sqrt((x-self.prev_x)**2 + (y-self.prev_y)**2)
        self.prev_x = x
        self.prev_y = y
        #print(dist)
        return dist

    def is_done(self):
        collision = False
        lidar_rays = self.lidar.getRangeImage()
        for i,ray in enumerate(lidar_rays):
            if ray > 4:
                lidar_rays[i] = 4
            if ray < 0.125:
                collision = True
        median_ray = statistics.median(lidar_rays)
        if(median_ray < 0.25):
            self.episode_score -= 10
            return True
        if collision == True:
            self.episode_score -= 10
            return True
        xAng, yAng, zAng, Angle = self.get_rotation()
        if (yAng < -0.9 or xAng < -0.9) and ( Angle > 3 or Angle < -3):
            print("CAR HAS FLIPPED")
            self.episode_score -= 10
            return True
            
        
        if self.completed_lap == True:
            self.episode_score += 100
            return True
        
        return False

    def solved(self):
        if len(self.episode_score_list) > 200:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-200:]) > 200:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        action = int(action[0])
        #print(action)
        if action == 0:
            self.set_velocity(0)
            self.set_steering_angle(0)
        elif action == 1:
            self.set_velocity(self.DRIVE_MAX_SPEED)
            self.set_steering_angle(0)
        elif action == 2:
            self.set_velocity(0)
            self.set_steering_angle(self.STEER_MAX_ANGLE)
        else:
            self.set_velocity(0)
            self.set_steering_angle(-self.STEER_MAX_ANGLE)
            

env = Learn2Drive()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n, 
                    use_cuda=True, batch_size=16, actor_lr = 0.0001, critic_lr = 0.00015)
#agent.load("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP2_Agents\\sp2_12")
#for param in agent.actor_net.parameters():
#    param.requires_grad = False
#for param in agent.critic_net.parameters():
#    param.requires_grad = False
#for param in agent.actor_net.parameters():
#    param.requires_grad = True
#for param in agent.critic_net.parameters():
#    param.requires_grad = True

solved = False
episode_count = 0
episode_limit = 50000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")
    agent.save("C:\\Users\\mcgra\\Documents\\Webots_Projects\\AR_RaceCar\\Actor-Critic_Agents\\Racer_02")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectAction")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()