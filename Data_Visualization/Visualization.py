import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as pltcol

class Datalogger:
    """
    This is the class for logging vehicle data in the F1TENTH Gym
    """
    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def __init__(self, conf):
        self.conf = conf                            # Current configuration for the gym based on the maps
        self.load_waypoints(conf)                   # Waypoints of the raceline
        self.vehicle_position_x = []                # Current vehicle position X (rear axle) on the map
        self.vehicle_position_y = []                # Current vehicle position Y (rear axle) on the map
        self.vehicle_position_heading = []          # Current vehicle heading on the map
        self.vehicle_velocity = []                  # Current vehicle velocity
        self.control_velocity = []                  # Desired vehicle velocity based on control calculation
        self.steering_angle = []                    # Steering angle based on control calculation
        self.lapcounter = []                        # Current vehicle velocity

    def logging(self, pose_x, pose_y, pose_theta, current_velocity, lap, control_veloctiy, control_steering):
        self.vehicle_position_x.append(pose_x)
        self.vehicle_position_y.append(pose_y)
        self.vehicle_position_heading.append(pose_theta)
        self.vehicle_velocity.append(current_velocity)
        self.control_velocity.append(control_veloctiy)
        self.steering_angle.append(control_steering)
        self.lapcounter.append(lap)

# Load the pickle file data
current_path = os.getcwd()
filename = current_path + '/datalogging.p'
file_to_read = open(filename, "rb")
data = pickle.load(file_to_read)

# Extract Raceline data
raceline_x = data.waypoints[:,[1]]
raceline_y = data.waypoints[:,[2]]
raceline_heading = data.waypoints[:,[3]]



###############################################################################################################
################################      Visualize Vehicle Data     ##############################################

###########   Plot driven path of vehicle for all laps    #######
plt.figure(1)
plt.plot(data.vehicle_position_x,data.vehicle_position_y,linestyle ='solid',linewidth=2, color = '#005293', label = 'Driven Path')
plt.plot(raceline_x,raceline_y,linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Path')
plt.axis('equal')
plt.xlabel ('X-Position on track')
plt.ylabel ('Y-Position on track')
plt.legend()
plt.title ('Vehicle Path: Driven Path vs. Raceline Path')


###########   Velocity of the vehicle    #######
plt.figure(2)
plt.plot(data.vehicle_velocity, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Veloctiy')
plt.xlabel ('Timesteps')
plt.plot(data.control_velocity, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Veloctiy')
plt.ylabel ('Velocity in m/s')
plt.legend()
plt.title ('Vehicle Velocity: Actual Velocity vs. Raceline Velocity')

###########   Heading of the Vehicle    #######
plt.figure(3)
plt.plot(data.vehicle_position_heading , linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
#plt.plot(raceline_heading, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Heading')
plt.xlabel ('Timesteps')
plt.ylabel ('Vehicle Heading')
plt.legend()
plt.title ('Vehicle Heading: Actual Heading vs. Raceline Heading')
plt.show()

###########   Steering Angle    #######
plt.figure(4)
plt.plot(data.steering_angle, linestyle ='solid',linewidth=2,color = '#005293', label = 'Actual Heading')
#plt.plot(raceline_heading, linestyle ='dashed',linewidth=2, color = '#e37222', label = 'Raceline Heading')
plt.xlabel ('Timesteps')
plt.ylabel ('Steering angle in degree')
plt.legend()
plt.title ('Steering angle')
plt.show()
