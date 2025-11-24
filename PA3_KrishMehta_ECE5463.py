"""
PA3 Pick-and-Place Joint-Space Trajectory Planning & PID Control Homework Assignment ECE 5463 - Krish Mehta

2-DOF planar RR pick-and-place (2 link)

IMPORTANT NOTE FOR SCRIPT: q refers to theta from equations document in Github repo (in the submitted report)

References:
    FOR EOM DERIVATONS:
 - Lecture 13 - Robot Dynamic-Lagrangian Method
 - Lecture 14 - Lagrange-Euler RP Manipulator Example
 - Lecture 15 - Lagrange-Euler Explicit Form
 - Lecture 16 - Feedback Control
 - Lecture 17 - Trajectory Tracking and Force Control
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Link parameters (in meters, m)
L1 = 0.5
L2 = 0.3

# Chosen link masses (in kg)
m1 = 1.0
m2 = 0.8

# Inertia about COM (center-of-mass) = (1/12)*m*L^2
I1 = (1.0/12.0) * m1 * L1**2
I2 = (1.0/12.0) * m2 * L2**2

# COM distances from joint
lc1 = L1 / 2.0
lc2 = L2 / 2.0

g = 9.81  # gravity (in m/s^2)

# Joint limits (in radians, need to convert)
joint_min = np.deg2rad([-90.0, -90.0])
joint_max = np.deg2rad([ 90.0,  90.0])

# Pick-and-place joint configurations (radians)
home = np.deg2rad(np.array([0.0,   0.0]))
pick = np.deg2rad(np.array([30.0, -20.0]))
place = np.deg2rad(np.array([45.0, -30.0]))

# Kinematics - forward kinematics
def fk(q):
# Return end-effector (x,y) for joint angles theta = [q1, q2]
    q1, q2 = q
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2)  # Taken from Lines 52-53 from my team's PA2 script (Python Group 6)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2)  # Same as above
    return np.array([x, y])

# Dynamics: M(q), C(q,q_d), G(q)
def M_matrix(q):
    q1, q2 = q
    M11 = I1 + I2 + m1*lc1**2 + m2*(L1**2 + lc2**2 + 2*L1*lc2*np.cos(q2))
    M12 = I2 + m2*(lc2**2 + L1*lc2*np.cos(q2))
    M22 = I2 + m2*lc2**2
    return np.array([[M11, M12],
                     [M12, M22]])

def C_matrix(q, q_d):
# Coriolis/centripetal matrix, q_d is q_desired
    q1, q2 = q
    q1_d, q2_d = q_d
    h = -m2 * L1 * lc2 * np.sin(q2)
    c11 = h * q2_d
    c12 = h * (q1_d + q2_d)
    c21 = -h * q1_d
    c22 = 0.0
    return np.array([[c11, c12],
                     [c21, c22]])

# Gravity vector
def G_vector(q):
    q1, q2 = q
    g1 = (m1*lc1 + m2*L1) * g * np.cos(q1) + m2*lc2 * g * np.cos(q1 + q2)
    g2 = m2*lc2 * g * np.cos(q1 + q2)
    return np.array([g1, g2])

# Cubic joint trajectory
def cubic_segment(q0, qf, T, t_array):
# Cubic polynomial interpolation from q0 to qf over times t_array in [0,T].
# Zero start & end velocities.
# Returns: qd (N,2), qd_dot (N,2), qd_ddot (N,2)
    q0 = np.array(q0)
    qf = np.array(qf)
    a0 = q0
    a1 = np.zeros_like(q0)
    a2 = (3.0/(T**2)) * (qf - q0)
    a3 = (-2.0/(T**3)) * (qf - q0)
    td = t_array[:, None]
    qd = a0 + a1*td + a2*(td**2) + a3*(td**3)
    qd_dot = a1 + 2*a2*td + 3*a3*(td**2)
    qd_ddot = 2*a2 + 6*a3*td
    return qd, qd_dot, qd_ddot

def build_full_trajectory(dt=0.02, t_move=2.0, t_pause=1.0):
# Builds full desired joint trajectories (Home to Pick, hold, Pick to Place, hold)
    t1 = np.arange(0.0, t_move, dt)  # citation for "arange" in PA3 report pdf file
    t2 = np.arange(0.0, t_pause, dt)
    t3 = np.arange(0.0, t_move, dt)
    t4 = np.arange(0.0, t_pause, dt)

    if t1.size == 0: t1 = np.array([0.0])
    if t2.size == 0: t2 = np.array([0.0])
    if t3.size == 0: t3 = np.array([0.0])
    if t4.size == 0: t4 = np.array([0.0])

    q1, q1d, q1dd = cubic_segment(home, pick, t_move, t1)
    q2 = np.tile(pick, (len(t2), 1)); q2d = np.zeros_like(q2); q2dd = np.zeros_like(q2)
    # np.tile repeats an array along specified dimensions, citation for it in PA3 report pdf file
    q3, q3d, q3dd = cubic_segment(pick, place, t_move, t3)
    q4 = np.tile(place, (len(t4), 1)); q4d = np.zeros_like(q4); q4dd = np.zeros_like(q4)

    q_des = np.vstack([q1, q2, q3, q4])
    qd_des = np.vstack([q1d, q2d, q3d, q4d])
    qdd_des = np.vstack([q1dd, q2dd, q3dd, q4dd])

    # timestamps
    t_seg1 = t1
    t_seg2 = t_move + t2
    t_seg3 = t_move + t_pause + t3
    t_seg4 = 2*t_move + t_pause + t4
    time = np.concatenate([t_seg1, t_seg2, t_seg3, t_seg4])

    return time, q_des, qd_des, qdd_des

# Controller: Computed-torque (inverse dynamics)
# Proportional and derivative gains tuned by trial and error to maximize best results for simulation
# Computed torque cancels nonlinear dynamics (M,C,G), therefore turning problem linear
# This allows straightforward tuning of Kp and Kd
Kp = np.diag([120.0, 80.0])
Kd = np.diag([18.0, 12.0])

def computed_torque(q, qd, q_des, qd_des, qdd_des):
# tau = M(q)*(qdd_des + Kd*(qd_des - qd) + Kp*(q_des - q)) + C(q,qd)*qd_des + G(q)
    e = q_des - q
    ed = qd_des - qd
    v = qdd_des + Kd.dot(ed) + Kp.dot(e)
    M = M_matrix(q)
    C = C_matrix(q, qd)
    G = G_vector(q)
    tau = M.dot(v) + C.dot(qd_des) + G
    return tau

# Dynamics integrator (RK4)
def dynamics_acc(qi, dqi, taui):
    M = M_matrix(qi)
    C = C_matrix(qi, dqi)
    G = G_vector(qi)
    return np.linalg.solve(M, taui - C.dot(dqi) - G)

def simulate(dt=0.02):
    time, q_des_traj, qd_des_traj, qdd_des_traj = build_full_trajectory(dt=dt)
    N = len(time)
    q = np.zeros((N,2))
    qd = np.zeros((N,2))
    tau_log = np.zeros((N,2))
    ee_log = np.zeros((N,2))

    # initial state
    q[0,:] = home.copy()
    qd[0,:] = np.zeros(2)

    for i in range(N-1):
        q_des = q_des_traj[i]
        qd_des = qd_des_traj[i]
        qdd_des = qdd_des_traj[i]

        tau = computed_torque(q[i], qd[i], q_des, qd_des, qdd_des)
        tau = np.clip(tau, -200.0, 200.0)
        tau_log[i] = tau

        # RK4 integration for second-order system
        state = np.hstack((q[i], qd[i]))
        h = dt

        def deriv(s, tau_local):
            qi = s[:2]
            dqi = s[2:]
            ddqi = dynamics_acc(qi, dqi, tau_local)
            return np.hstack((dqi, ddqi))

        k1 = deriv(state, tau)
        k2 = deriv(state + 0.5*h*k1, tau)
        k3 = deriv(state + 0.5*h*k2, tau)
        k4 = deriv(state + h*k3, tau)

        state_next = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q[i+1] = state_next[:2]
        qd[i+1] = state_next[2:]
        ee_log[i] = fk(q[i])

        # Enforcing the joint limits
        q[i+1] = np.clip(q[i+1], joint_min, joint_max)

    ee_log[-1] = fk(q[-1])
    tau_log[-1] = tau_log[-2]
    return time, q, qd, tau_log, ee_log, q_des_traj, qd_des_traj, qdd_des_traj

# Plot functions
def plot_results(time, q_log, q_des_traj, ee_log):
    # End-effector path with point masses for Pick/Place
    p_home = fk(home); p_pick = fk(pick); p_place = fk(place)
    plt.figure(figsize=(7,7))
    plt.plot(ee_log[:,0], ee_log[:,1], '-b', label='End-effector path')
    plt.scatter([p_home[0]],[p_home[1]], label='Home', c='k')
    plt.scatter([p_pick[0]],[p_pick[1]], label='Pick (point mass)', c='r', s=100)
    plt.scatter([p_place[0]],[p_place[1]], label='Place (point mass)', c='g', s=100)
    plt.gca().set_aspect('equal', 'box') # gca() gets the current axes in order to set the aspect
    # ratio of the plot; without it, the robot's path could look stretched - cited in PA3 report
    plt.title('End-effector path (XY) with Pick/Place point masses')
    plt.xlabel('x [m]'); plt.ylabel('y [m]'); plt.grid(True); plt.legend()
    plt.show()

    # EE coordinates vs time to show pauses between movements between locations (velocity control)
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(time, ee_log[:,0]); plt.ylabel('EE x [m]')
    plt.subplot(2,1,2)
    plt.plot(time, ee_log[:,1]); plt.ylabel('EE y [m]'); plt.xlabel('Time [s]')
    plt.suptitle('End-effector coordinates vs time (pauses visible)')
    plt.tight_layout()
    plt.show()

# Animation
def animate_motion(q_log, interval_ms=8):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    rmax = L1 + L2 + 0.1
    ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
    ax.grid(True); ax.set_title('2-DOF RR Pick-and-Place (Computed Torque)')

    p_pick = fk(pick); p_place = fk(place)
    ax.plot(p_pick[0], p_pick[1], 'rx', markersize=10, label='Pick')
    ax.plot(p_place[0], p_place[1], 'gx', markersize=10, label='Place')
    ax.legend()

    line, = ax.plot([], [], 'o-', lw=4)

    def update(i):
        q1, q2 = q_log[i]
        x1 = L1*np.cos(q1); y1 = L1*np.sin(q1)
        x2 = x1 + L2*np.cos(q1 + q2); y2 = y1 + L2*np.sin(q1 + q2)
        line.set_data([0, x1, x2], [0, y1, y2])
        return line,

    ani = FuncAnimation(fig, update, frames=len(q_log), interval=interval_ms, blit=True)
    plt.show()

# Main
if __name__ == "__main__":
    dt = 0.02
    time, q_log, qd_log, tau_log, ee_log, q_des_traj, qd_des_traj, qdd_des_traj = simulate(dt=dt)

    # Plots
    plot_results(time, q_log, q_des_traj, ee_log)

    # Animation
    animate_motion(q_log)

    # Steady-state error at final (last 0.2 s)
    final_window = 0.2
    idxs = np.where(time >= time[-1] - final_window)[0]
    final_des = q_des_traj[-1]
    final_act = q_log[-1]
    err = np.abs(final_des - final_act)
    err_deg = np.rad2deg(err)
    print(f"Final steady-state error (deg): joint1 = {err_deg[0]:.6f}, joint2 = {err_deg[1]:.6f}")
    print(f"Max final error (deg) = {np.max(err_deg):.6f}")