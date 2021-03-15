import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import pandas as pd
import math
import matplotlib.pyplot as plt

from numba import njit

""" 
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

#@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

@njit(fastmath=False, cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle



class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

    def load_waypoints(self, conf):
        # load waypoints
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, veloctiy, vgain):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle

class StanleyPlanner:
    """
    Front-Wheel Feedback Controller (Stanley)
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb                 # Wheelbase of the vehicle
        self.conf = conf                    # Current configuration for the gym based on the maps
        self.load_waypoints(conf)           # Waypoints of the raceline
        self.ind_old = 0                    # Current Waypoint index

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def calc_theta_and_ef(self, vehicle_state, waypoints):
        " calc theta and ef"
        " Theta is the heading of the car, this heading must be minimized"
        " ef = crosstrack error/The distance from the optimal path"
        " ef = lateral distance in frenet frame (front wheel)"

        ############# Calculate closest point to the front axle based on minimum distance calculation ################

        # Extract the raceline specific waypoints x,y, heading, velocity
        wpts_x = self.waypoints[:, [1]]
        wpts_y = self.waypoints[:, [2]]
        wpts_yaw = self.waypoints[:, [3]]
        wpts_veloctiy = self.waypoints[:, [5]]
        wpts_x2 = wpts_x.tolist()
        wpts_y2 = wpts_y.tolist()
        wpts_yaw2 = wpts_yaw.tolist()

        # Calculate Position of the front axle
        fx = vehicle_state[0] + self.wheelbase * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase * math.sin(vehicle_state[2])

        plt.cla()
        plt.plot(wpts_x, wpts_y, color='gray', linewidth=2.0)  # Plot Raceline

        # Calculate the Distances from the front axle to all the waypoints
        dx = [fx - x for x in wpts_x2]
        dy = [fy - y for y in wpts_y2]

        # Find the target index for the correct waypoint by finding the index with the lowest value
        target_index = int(np.argmin(np.hypot(dx, dy)))
        target_index = max(self.ind_old, target_index)
        self.ind_old = max(self.ind_old, target_index)

        ############################# Speed up calculation ###################################################
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        position_front_axle = np.array([fx, fy])
        nearest_point_front, nearest_dist, t, i = nearest_point_on_trajectory(position_front_axle, wpts)

        ###################  Calculate the current Cross-Track Error ef in [m]   ################

        # Project crosstrack error onto front axle vector
        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - math.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - math.pi / 2.0)]])

        vec_target_2_front = np.array([dx[target_index], dy[target_index]])

        # Caculate the cross-track error ef by
        ef = np.dot(vec_target_2_front.T, front_axle_vec_rot_90)
        ef2 = np.dot(nearest_point_front.T, front_axle_vec_rot_90)

        #######################################################################################
        ################## DEBUG - Plotting the points" #############################


        plt.plot(dx[target_index][0], dy[target_index][0], '.r')    #Plot Current vehicle position - rear axle
        plt.plot(position_front_axle[0],position_front_axle[1], '.b')   #Plot Current vehicle position - front axle
        plt.plot(nearest_point_front[0],nearest_point_front[1], '.c')   # Plot Nearest Point to rear axel -BILLY calculation
        ################## DEBUG - Plotting the points" #############################
        #######################################################################################


        #############  Calculate the heading error theta_e  normalized to an angle to [-pi, pi]     ##########

        # Extract heading on the raceline - different COSY in raceline path so add of pi/2 = 90 degrees
        theta_raceline = wpts_yaw2[target_index][0] + math.pi/2

        # Calculate the heading error by taking the difference between current and goal + Normalize the angles
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # Calculate the target Veloctiy for the desired state
        goal_veloctiy = wpts_veloctiy[target_index][0]

        return theta_e, ef, target_index, goal_veloctiy

    def controller(self, vehicle_state, waypoints, vgain):
        " Front Wheel Feedback Controller to track the path "
        " Based on the heading error theta_e and the crosstrack error ef we calculate the steering angle"
        " Returns the optimal steering angle delta is P-Controller with the proportional gain k"

        k_path =5.2                 # Proportional gain for path control
        k_veloctiy = vgain           # Proportional gain for speed control, defined globally in the gym
        theta_e, ef, target_index, goal_veloctiy = self.calc_theta_and_ef(vehicle_state, waypoints)

        # Caculate steering angle based on the cross track error to the front axle in [rad]
        cte_front = math.atan2(k_path * ef, vehicle_state[3])

        # Calculate final steering angle/ control input "delta" in [rad]: Crosstrack error ef + heading error
        delta = cte_front + theta_e

        # Calculate final speed control input in [m/s]:
        #speed_diff = k_veloctiy * (goal_veloctiy-velocity)
        speed = k_veloctiy * goal_veloctiy

        return delta, speed

    def plan(self, pose_x, pose_y, pose_theta, velocity, vgain):
        #Define a numpy array that includes the current vehicle state: Pose + Veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        #Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(vehicle_state, self.waypoints, vgain)

        return speed,steering_angle


if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.75}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Creating the Motion planner object that is used in the F1TENTH Gym
    planner = StanleyPlanner(conf, 0.17145 + 0.15875)

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], work['vgain'])

        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human_fast')

    print("Racetrack")
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)