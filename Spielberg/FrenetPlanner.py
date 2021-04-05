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
import cubic_spline_planner
import trajectory_planning_helpers.path_matching_local as tph
import trajectory_planning_helpers.path_matching_global as tph

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

class PurePursuitPlanner:
    """
    This is the PurePursuit ALgorithm that is traccking the desired path. In this case we are following the curvature
    optimal raceline.
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.vsv" that include the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, path):
        # Find the current waypoint on the map and calculate the lookahead point for the controller
        # wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        # Create waypoints based on the current frenet path
        wpts = np.vstack((np.array(path.x), np.array(path.y))).T

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

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain, path):
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, path)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


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
        self.c_d = 0.0                      # current lateral position in the Frenet Frame [m]
        self.c_d_d = 0.0                    # current lateral speed in the Frenet Frame [m/s]
        self.c_d_dd = 0.0                   # current lateral acceleration in the Frenet Frame [m/s]
        self.s0 = 0.0                       # current course position s in the Frenet Frame
        self.calcspline = 0
        self.csp = 0
        self.debug_count = 0                # DEBUG - Counts
        self.debug_array1 = []              # DEBUG - array for saving numbers
        self.debug_array2 = []              # DEBUG - array for saving numbers
        self.debug_array3 = []  # DEBUG - array for saving numbers
        self.debug_array4 = []  # DEBUG - array for saving numbers

    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """

        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)


    def check_collision(self, fp, ob):
        ROBOT_RADIUS = 0.5                  # robot radius [m]

        for i in range(len(ob[:, 0])):
            d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
                 for (ix, iy) in zip(fp.x, fp.y)]

            collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False

        return True

    def check_paths(self, fplist,ob):
        MAX_SPEED = 12.0                    # maximum speed [m/s]
        MAX_ACCEL = 8.0                     # maximum acceleration [m/ss]
        MAX_CURVATURE = 1.0                 # maximum curvature [1/m]

        ok_ind = []
        for i, _ in enumerate(fplist):
            if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                continue
            elif any([abs(a) > MAX_ACCEL for a in
                      fplist[i].s_dd]):  # Max accel check
                continue
            elif any([abs(c) > MAX_CURVATURE for c in
                      fplist[i].c]):  # Max curvature check
                continue
            elif not self.check_collision(fplist[i], ob):
                continue

            ok_ind.append(i)

        return [fplist[i] for i in ok_ind]

    def calc_frenet_paths(self, c_speed, c_d, c_d_d, c_d_dd, s0):
        # Parameter
        MAX_ROAD_WIDTH = 1.00       # maximum road width [m]
        D_ROAD_W = 0.50            # road width sampling length [m]
        MAX_T = 1.5                 # max prediction time [m]
        MIN_T = 0.5                 # min prediction time [m]
        DT = 0.2                    # Sampling time in s
        TARGET_SPEED = 8.0          # Target speed in [m/s]
        D_T_S = 1.0                 # target speed sampling length [m/s]
        N_S_SAMPLE = 1              # sampling number of target speed

        # cost weights
        K_J = 0.1                   # Weights for Jerk
        K_T = 0.1                   # Weights for Time
        K_D = 100.0                   # Weights for
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

                #Calculate Later Position
                fp.t = [t for t in np.arange(0.0, Ti, DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                # Calculate first derivative of the position: Lateral Veloctiy
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                # Calculate second derivative of the position: Lateral Acceleration
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                # Calculate third derivative of the position: Lateral Jerk
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                    TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                    tfp = copy.deepcopy(fp)
                    lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                    #Calculate longitudinal position
                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    # Calculate first derivative of longitudinal position: longitudinal veloctiy
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    # Calculate second derivative of longitudinal position: longitudinal acceleration
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    # Calculate third derivative of longitudinal position: longitudinal jerk
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                    # Calculate Lateral Costs: Influence Jerk Lat + Influence Time + Influence Distance from optimal path
                    tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                    # Calculate Lateral Costs: Influence Jerk Long + Influence Time + Influence Difference Speed
                    tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                    # Calculate final cost of the frenet Path: Weight_Lateral * Costs_Lateral + Weight_Longitudinal * Costs_Longitudinal
                    tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def calc_global_paths(self, fplist, csp, vehicle_state):
        # Calculating the maximal s-value to make the s=0 transition on the star/finish line
        s_max = max(self.waypoints[:,[0]])

        # For loop for calculation the global x and y positions
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                if fp.s[i] > s_max[0]:
                    fp.s[i] = fp.s[i] - s_max[0]

                ix, iy = csp.calc_position(fp.s[i], s_max[0])

                if ix is None:
                    break
                i_yaw = csp.calc_yaw(fp.s[i], s_max[0])
                di = fp.d[i]
                fx = ix - di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy - di * math.sin(i_yaw + math.pi / 2.0)
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

        self.debug_count = self.debug_count +1
        if self.debug_count == 89:
            test = 0
        return fplist

    def path_planner(self, vehicle_state,  obstacles):

        # Calculate the cubic spline of the raceline path and create csp object - create it once!
        if self.calcspline == 0:
            self.csp = cubic_spline_planner.Spline2D(self.waypoints[:,1], self.waypoints[:,2])
            self.calcspline = 1

        # Get current position S and distance d to the global raceline
        state = np.stack((vehicle_state[0], vehicle_state[1]), axis=0)
        traj = np.stack((self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2]), axis=-1)
        self.s0, self.c_d = tph.path_matching_global(traj,state)

        # Calculate the optimal paths in the frenet frame
        fplist = self.calc_frenet_paths(vehicle_state[3], self.c_d, self.c_d_d, self.c_d_dd, self.s0)

        # Calculate the one optimal path based closest to the global path (raceline)
        fplist = self.calc_global_paths(fplist, self.csp, vehicle_state)

        # Collision Check: Check if there are obstacles in the way of the path
        #fplist = self.check_paths(fplist, obstacles)

        # Find the path with the minimum cost = optimal path to drive
        min_cost = float("inf")
        best_path = None
        for fp in fplist:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        # Update additional paramter
        self.c_d_d = best_path.d_d[1]
        self.c_d_dd = best_path.d_dd[1]


        ###########################################
        #                    DEBUG
        ##########################################

        debugplot=0
        if debugplot == 1:
            plt.cla()
            plt.axis([-40, 2, -10, 10])
            plt.plot(self.waypoints[:,[1]], self.waypoints[:,[2]], linestyle='solid', linewidth=2, color='#005293')
            plt.plot(vehicle_state[0],vehicle_state[1], marker='o', color='red')
            for fp in fplist:
                plt.plot(fp.x, fp.y,linestyle ='dashed',linewidth=2, color = '#e37222')

            plt.plot(best_path.x,best_path.y,linestyle ='dotted',linewidth=3, color = 'green')
            plt.pause(0.001)
            plt.axis('equal')


        ###########################################
        #                    DEBUG
        ###########################################

        return best_path

    def plan(self, pose_x, pose_y, pose_theta, velocity, vgain, timestep):
        # Define a numpy array that includes the current vehicle state: x-position,y-position, theta, veloctiy
        vehicle_state = np.array([pose_x, pose_y, pose_theta, velocity])

        # Detect Obstacles on the track
        obstacles = np.array([[20.0, 10.0],[30.0, 6.0]])

        # Calculate the optimal path in the frenet frame
        path = self.path_planner(vehicle_state, obstacles)

        # Calculate the steering angle and the speed in the controller
        speed, steering_angle = controller.plan(pose_x, pose_y, pose_theta, 0.9, 0.5, path)

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
    controller = PurePursuitPlanner(conf, 0.17145 + 0.15875)

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