import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
from numba import njit
import matplotlib.pyplot as plt
import pickle
import copy

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

class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt

class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

class FrenetPlaner:
    """
    Frenet optimal trajectory generator

    References:
    - [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
    (https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

    - [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
    (https://www.youtube.com/watch?v=Cj6tAQe7UCY)

    """

    def __init__(self, conf, env, wb):
        self.wheelbase = wb                 # Wheelbase of the vehicle
        self.conf = conf                    # Current configuration for the gym based on the maps
        self.env = env                     # Current environment parameter
        self.load_waypoints(conf)           # Waypoints of the raceline
        self.max_reacquire = 20.
        self.vehicle_control_e_cog = 0       # e_cg: lateral error of CoG to ref trajectory
        self.vehicle_control_theta_e = 0     # theta_e: yaw error to ref trajectory
        self.c_d = 2.0                      # current lateral position [m]
        self.c_d_d = 0.0                    # current lateral speed [m/s]
        self.c_d_dd = 0.0                   # current lateral acceleration [m/s]
        self.s0 = 0.0                       # current course position

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

    def path_controller(self, vehicle_state, waypoints, vgain, timestep):
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

    def calc_frenet_paths(self, c_speed, c_d, c_d_d, c_d_dd, s0):
        # Parameter
        MAX_ROAD_WIDTH = 1.1/2      # maximum road width [m]
        D_ROAD_W = 0.1              # road width sampling length [m]
        MAX_T = 5.0                 # max prediction time [m]
        MIN_T = 4.0                 # min prediction time [m]
        DT = 0.1                    # Sampling time in s
        TARGET_SPEED = 8.0          # Target speed in [m/s]
        D_T_S = 1.0                 # target speed sampling length [m/s]
        N_S_SAMPLE = 4              # sampling number of target speed

        # cost weights
        K_J = 0.1
        K_T = 0.1
        K_D = 1.0
        K_LAT = 1.0
        K_LON = 1.0

        frenet_paths = []

        # generate path to each offset goal
        for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

            # Lateral motion planning
            for Ti in np.arange(MIN_T, MAX_T, DT):
                fp = FrenetPath()

                # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
                lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                    TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                    tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                    tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def calc_global_paths(self, fplist, csp):
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # calc yaw and ds
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # calc curvature
            for i in range(len(fp.yaw) - 1):
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

        return fplist

    def check_paths(self, speed, c_d, c_d_d, c_d_dd, s0):
        path = 0

        return path

    def path_planner(self, vehicle_state, waypoints, timestep):

        #Calculate the optimal path in the frenet frame
        fplist = self.calc_frenet_paths(vehicle_state[3], self.c_d, self.c_d_d, self.c_d_dd, self.s0)

        # Calculate the global path based on the waypoints
        fplist = self.calc_global_paths(fplist, csp)

        # Check if the obstacles are in the way of the path
        fplist = self.check_paths(fplist, ob)

        # find minimum cost path
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        best_path

        return best_path

    def plan(self, pose_x, pose_y, pose_theta, velocity, vgain, timestep):
        #Define a numpy array that includes the current vehicle state: x,y, theta, veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        #Calculate the optimal path in the frenet frame
        path = self.path_planner(vehicle_state, self.waypoints, timestep)

        # Calculate the steering angle and the speed in the controller
        steering_angle, speed = self.controller(vehicle_state, self.waypoints, vgain, timestep)

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
    planner = FrenetPlaner(conf, env, 0.17145 + 0.15875)

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
        pickle.dump(logging, open("datalogging.p", "wb"))
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)