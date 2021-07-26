import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
from numba import njit
import matplotlib.pyplot as plt
import pickle

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
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

@njit(fastmath=False, cache=True)
def SolveLQRProblem(A, B, Q, R, tolerance, max_num_iteration):
    """
        Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
    iteratively calculating feedback matrix K
    :param A: matrix_a_
    :param B: matrix_b_
    :param Q: matrix_q_
    :param R: matrix_r_
    :param tolerance: lqr_eps
    :param max_num_iteration: max_iteration
    :return: feedback matrix K
    """
    M = np.zeros((Q.shape[0], R.shape[1]))

    AT = A.T
    BT = B.T
    MT = M.T

    P = Q
    num_iteration = 0
    diff = math.inf

    # First, try to solve a discrete time_Algebraic Riccati equation (DARE)
    while num_iteration < max_num_iteration and diff > tolerance:
        num_iteration += 1
        P_next = AT @ P @ A - (AT @ P @ B + M) @ \
                 np.linalg.pinv(R + BT @ P @ B) @ (BT @ P @ A + MT) + Q

        # check the difference between P and P_next
        # diff = (abs(P_next - P)).max()
        diff = np.abs(np.max(P_next - P))
        P = P_next

    # Compute the LQR gain by iteratively calculating feedback matrix K
    K = np.linalg.pinv(BT @ P @ B + R) @ (BT @ P @ A + MT)

    return K

@njit(fastmath=False, cache=True)
def UpdateMatrix(vehicle_state,state_size,timestep,wheelbase):
    """
    calc A and b matrices of linearized, discrete system.
    :return: A, b
    """

    #Current vehicle velocity
    v = vehicle_state[3]

    #Initialization of the time discrete A matrix
    matrix_ad_ = np.zeros((state_size, state_size))

    matrix_ad_[0][0] = 1.0
    matrix_ad_[0][1] = timestep
    matrix_ad_[1][2] = v
    matrix_ad_[2][2] = 1.0
    matrix_ad_[2][3] = timestep

    # b = [0.0, 0.0, 0.0, v / L].T
    matrix_bd_ = np.zeros((state_size, 1))  # time discrete b matrix
    matrix_bd_[3][0] = v / wheelbase

    return matrix_ad_, matrix_bd_

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

class LQR_Kinematic_Planner:
    """
    Lateral Controller using LQR
    """

    def __init__(self, conf, env, wb):
        self.wheelbase = wb                 # Wheelbase of the vehicle
        self.conf = conf                    # Current configuration for the gym based on the maps
        self.env = env                     # Current environment parameter
        self.load_waypoints(conf)           # Waypoints of the raceline
        self.max_reacquire = 20.
        self.vehicle_control_e_cog = 0       # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0     # theta_e: yaw error to ref trajectory

    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """

        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def calc_control_points(self, vehicle_state, waypoints):
        """
        Calculate all the errors to trajectory frame + find desired curvature and heading
        calc theta and ef
        Theta is the heading of the car, this heading must be minimized
        ef = crosstrack error/The distance from the optimal path/ lateral distance in frenet frame (front wheel)
        """

        ############# Calculate closest point to the front axle based on minimum distance calculation ################
        # Calculate Position of the front axle of the vehicle based on current position
        fx = vehicle_state[0] + self.wheelbase/2 * math.cos(vehicle_state[2])
        fy = vehicle_state[1] + self.wheelbase/2 * math.sin(vehicle_state[2])
        position_front_axle = np.array([fx, fy])

        # Find target index for the correct waypoint by finding the index with the lowest distance value/hypothenuses
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point_front, nearest_dist, t, target_index = nearest_point_on_trajectory(position_front_axle, wpts)

        # Calculate the Distances from the front axle to all the waypoints
        distance_nearest_point_x= fx - nearest_point_front[0]
        distance_nearest_point_y = fy - nearest_point_front[1]
        vec_dist_nearest_point = np.array([distance_nearest_point_x, distance_nearest_point_y])

        ###################  Calculate the current Cross-Track Error ef in [m]   ################
        # Project crosstrack error onto front axle vector
        front_axle_vec_rot_90 = np.array([[math.cos(vehicle_state[2] - math.pi / 2.0)],
                                          [math.sin(vehicle_state[2] - math.pi / 2.0)]])

        #vec_target_2_front = np.array([dx[target_index], dy[target_index]])

        # Caculate the cross-track error ef by
        ef = np.dot(vec_dist_nearest_point.T, front_axle_vec_rot_90)

        #############  Calculate the heading error theta_e  normalized to an angle to [-pi, pi]     ##########
        # Extract heading for the optimal raceline
        # BE CAREFUL: If your raceline is based on a different coordinate system you need to -+ pi/2 = 90 degrees
        theta_raceline = waypoints[target_index][3]

        # Calculate the heading error by taking the difference between current and goal + Normalize the angles
        theta_e = pi_2_pi(theta_raceline - vehicle_state[2])

        # Calculate the target Veloctiy for the desired state
        goal_veloctiy = waypoints[target_index][5]

        #Find Reference curvature
        kappa_ref = self.waypoints[target_index][4]

        #Saving control errors
        self.vehicle_control_e_cog = ef[0]
        self.vehicle_control_theta_e = theta_e

        return theta_e, ef[0], theta_raceline, kappa_ref, goal_veloctiy

    def controller(self, vehicle_state, waypoints, vgain, timestep):
        """
        ComputeControlCommand calc lateral control command.
        :param vehicle_state: vehicle state
        :param ref_trajectory: reference trajectory (analyzer)f
        :return: steering angle (optimal u), theta_e, e_cg
        """

        ##### Setup and initialize the LQR parameter
        state_size = 4
        matrix_q = [0.80, 0.0, 1.2, 0.0]
        matrix_r = [1.0]
        max_iteration = 150
        eps = 0.001

        # Saving lateral error and heading error from previous timestep
        e_cog_old = self.vehicle_control_e_cog
        theta_e_old = self.vehicle_control_theta_e

        # Calculating current errors and reference points from reference trajectory
        theta_e, e_cog, yaw_ref, k_ref, v_ref = self.calc_control_points(vehicle_state,waypoints)

        #Update the calculation matrix based on the current vehicle state
        matrix_ad_, matrix_bd_ = UpdateMatrix(vehicle_state, state_size,timestep,self.wheelbase)

        ##################  Solving the LQR problem
        matrix_r_ = np.diag(matrix_r)           # Extract the diagonal array from the R Matrix
        matrix_q_ = np.diag(matrix_q)           # Extract the diagonal array from the Q Matrix

        # Calculating the optimal gain matrix K, given a state-space model for the plant and weighting matrices Q, R
        matrix_k_ = SolveLQRProblem(matrix_ad_, matrix_bd_, matrix_q_,
                                         matrix_r_, eps, max_iteration)

        # Create the state vector (4x1): x = [e_cog, dot_e_cog, theta_e, dot_theta_e]
        matrix_state_ = np.zeros((state_size, 1))                    # Initialize State vectors
        matrix_state_[0][0] = e_cog                                  # Current lateral distance e_cog to optimal path
        matrix_state_[1][0] = (e_cog - e_cog_old) / timestep         # Derivative of e_cog
        matrix_state_[2][0] = theta_e                                # Current heading difference to optimal path
        matrix_state_[3][0] = (theta_e - theta_e_old) / timestep     # Derivative of theta_e

        # Calculate feedback steering angle
        # Input vector u = [delta], matrix_k * state ectors
        steer_angle_feedback = (matrix_k_ @ matrix_state_)[0][0]

        # Calculate feed forward control term to decrease the steady error
        steer_angle_feedforward = k_ref * self.wheelbase

        # Calculate final steering angle in [rad]
        steer_angle = steer_angle_feedback + steer_angle_feedforward

        # Calculate final speed control input in [m/s]:
        speed = v_ref * vgain

        return steer_angle, speed

    def plan(self, pose_x, pose_y, pose_theta, velocity, vgain, timestep):
        #Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        #Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(vehicle_state, self.waypoints, vgain,timestep)

        return speed,steering_angle


if __name__ == '__main__':

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.80}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # Creating the Motion planner object that is used in the F1TENTH Gym
    planner = LQR_Kinematic_Planner(conf, env, 0.17145 + 0.15875)

    # Creating a Datalogger object that saves all necessary vehicle data
    logging = Datalogger(conf)

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], work['vgain'],env.timestep)

        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human_fast')

        if conf_dict['logging'] == 'True':
            logging.logging(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], obs['lap_counts'],speed, steer)

    if conf_dict['logging'] == 'True':
        pickle.dump(logging, open("../Data_Visualization/datalogging.p", "wb"))
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)