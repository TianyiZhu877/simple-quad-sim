import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
import matplotlib.pyplot as plt
from neural_fly.utils import save_data, load_data

def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12

IDX_P_D_X = 0
IDX_P_D_Y = 1
IDX_P_D_Z = 2
IDX_V_D_X = 3
IDX_V_D_Y = 4
IDX_V_D_Z = 5
IDX_CTRL_INPUT_MAX = 6

NUM_ROTORS = 4

class Robot:
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self):
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])

        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.omega_motors = np.array([0.0, 0.0, 0.0, 0.0])
        self.state = self.reset_state_and_input(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0

        # self.times = [self.time]
        # self.states = self.state[np.newaxis, :]
        self.times = []
        self.states = np.empty((0,NO_STATES))
        self.references = np.empty((0,IDX_CTRL_INPUT_MAX))
        self.ctrl_outputs = np.empty((0,NUM_ROTORS))

    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    def update(self, omegas_motor, dt, f = np.array([0,0,0])):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])
        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) + 1/self.m * f
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt

        self.states = np.vstack([self.states, self.state])
        self.times.append(self.time)

    def control(self, p_d_I, v_d_I = np.zeros(3)):
        
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        # Position controller.
        k_p = 1.0
        k_d = 10.0
        v_r = - k_p * (p_I - p_d_I)
        a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81])
        f = self.m * a
        f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f
        thrust = np.max([0, f_b[2]])
        
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b) # + self.J @ omega_ref_dot
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))

        # omega_motor = np.array([-50, 10,20, 10])

        self.references = np.vstack([self.references, np.concatenate([p_d_I, v_d_I])])
        self.ctrl_outputs = np.vstack([self.ctrl_outputs, omega_motor])

        return omega_motor

    def get_state_dict(self):
        result = dict()
        result['t'] = np.array(self.times)
        result['pwm'] = self.ctrl_outputs

        field_names_list = [['p', 'v', 'q', 'w'], ['p_d','v_d']]
        field_idxs_list = [[IDX_POS_X, IDX_VEL_X, IDX_QUAT_W, IDX_OMEGA_X, NO_STATES], [IDX_P_D_X, IDX_V_D_X, IDX_CTRL_INPUT_MAX]]
        self_logs_list = [self.states, self.references]

        for field_names, field_idxs, self_logs in zip(field_names_list, field_idxs_list, self_logs_list):
            for i,field_name in enumerate(field_names):
                # range_min = field_idxs[i]
                # range_max = field_idxs[i+1] if i<len(field_idxs)-1 else NO_STATES
                result[field_name] = self_logs[:, field_idxs[i]:field_idxs[i+1]]

        return result


PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_pos_full_quadcopter(quad):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad



def control_propellers(quad):
    t = quad.time
    T = 1.5
    r = 2*np.pi * t / T


    
    F_wind = np.zeros(3)

    # if t > 1.0:
    #     print("Wind is blowing!")
    #     F_wind = np.array([-10.0, 0, 0])
        
    omega_w = np.array([1,1,1.1])
    F0_w = np.array([10, 6, 2])/3
    phi_w = np.array((1,2,3))
    phi_w = np.array((3,1,2))
    F_wind = F0_w*np.sin(omega_w*t+phi_w)

    # p_d_I = np.array([np.cos(r/2), np.sin(r), 0.0])
    p_d_I = np.array([1.0, 0.0, 0.0])

    prop_thrusts = quad.control(p_d_I = p_d_I)
    # Note: for Hover mode, just replace the desired trajectory with [1, 0, 1]
    quad.update(prop_thrusts, dt, f = F_wind)

def main():
    quadcopter = Robot()
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter)
        return get_pos_full_quadcopter(quadcopter)

    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

    # plottings:
    plt.plot(quadcopter.states[:, IDX_POS_X], quadcopter.states[:, IDX_POS_Y], label = 'actual')
    plt.plot(quadcopter.references[:, IDX_P_D_X], quadcopter.references[:, IDX_P_D_Y], label = 'reference', marker='x')
    plt.xlabel('p_x')
    plt.ylabel('p_y')

    # plt.plot(quadcopter.times, quadcopter.states[:, IDX_OMEGA_X], label = 'w_x')
    # plt.plot(quadcopter.times, quadcopter.states[:, IDX_OMEGA_Y], label = 'w_y')

    # plt.xlabel('t')
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.legend()
    plt.show()

    # print(list(quadcopter.get_state_dict().keys()))

    filename = input("save data, n for no: ")
    if  filename != 'n' and filename != 'q':
        save_data([quadcopter.get_state_dict()], 'neural_fly/data/hw2', filename, fields = list(quadcopter.get_state_dict().keys()))
    # print(quadcopter.body_frame)

if __name__ == "__main__":
    main()
