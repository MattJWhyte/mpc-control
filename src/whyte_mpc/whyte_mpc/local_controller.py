#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
# Python
from pynput.keyboard import Listener, Key, Controller
import matplotlib.pyplot as plt

# ROS
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry
import numpy as np
import time
import shutil
import os
import threading

from casadi import *


class LocalController(Node):


    def __init__(self):
        super().__init__("local_controller")
        self.cone_sub = self.create_subscription(
        ConeArrayWithCovariance,
        '/fusion/cones',
        self.cone_callback,
        5)
        self.target_x = 0
        self.target_y = 0

        self.ack_pub = self.create_publisher(AckermannDriveStamped, "/cmd", 1)
        self.a = [0.1]
        self.d = [0.0]

        self.state_hist = []

        self.pose_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            5)
        self.state = [0.01,0.01,0.01,0.01]
        self.x1_opt = []
        self.x2_opt = []

    def cone_callback(self, msg):
        Bx = []
        By = []
        Yx = []
        Yy = []
        for c in msg.blue_cones:
            Bx.append(c.point.x)
            By.append(c.point.y)
        for c in msg.yellow_cones:
            Yx.append(c.point.x)
            Yy.append(c.point.y)

        idx_B = None
        idx_Y = None

        if len(Bx) > 0:
            Bx, By = self.sort_cones(Bx, By)
            idx_B = min(1,len(Bx)-1)

        if len(Yx) > 0:
            Yx, Yy = self.sort_cones(Yx, Yy)
            idx_Y = min(1,len(Yx)-1)

        if idx_B is not None and idx_Y is not None:
            self.target_x = (Bx[idx_B] + Yx[idx_Y]) / 2.0
            self.target_y = (By[idx_B] + Yy[idx_Y]) / 2.0
        elif idx_B is not None:
            self.target_x = Bx[idx_B]
            self.target_y = By[idx_B]

            dx = self.target_x - Bx[0]
            dy = self.target_y - By[0]
            l = np.sqrt(dx**2+dy**2)
            dx /= l
            dy /= l
            dx_P = dy
            dy_P = -dx
            self.target_x += dx_P*2.5
            self.target_y += dy_P*2.5

        elif idx_Y is not None:
            self.target_x = Yx[idx_Y]
            self.target_y = Yy[idx_Y]

            dx = self.target_x - Bx[0]
            dy = self.target_y - By[0]
            l = np.sqrt(dx**2+dy**2)
            dx /= l
            dy /= l
            dx_P = -dy
            dy_P = dx
            self.target_x += dx_P*2.5
            self.target_y += dy_P*2.5
        else:
            self.target_x = 0.0
            self.target_y = 0.0


    def sort_cones(self, xs, ys):
        n = len(xs)
        def d(x1,y1,x2=0,y2=0):
            return np.sqrt((x1-x2)**2+(y1-y2)**2)
        def min_idx(x,y,exclude=[]):
            min_i = 0
            while min_i in exclude:
                min_i += 1
            min_d = d(xs[min_i],ys[min_i],x,y)
            for i in range(1,n):
                if i in exclude:
                    continue
                x_d = d(xs[i],ys[i],x,y)
                if x_d < min_d:
                    min_d = x_d
                    min_i = i
            return min_i
        path_idx = [min_idx(0,0)]
        path_xs = [xs[path_idx[0]]]
        path_ys = [ys[path_idx[0]]]
        for _ in range(1,n):
            last_idx = path_idx[-1]
            path_idx.append(min_idx(xs[last_idx], ys[last_idx],exclude=path_idx))
            path_xs.append(xs[path_idx[-1]])
            path_ys.append(ys[path_idx[-1]])
        return path_xs,path_ys


    # NEED TO VERIFY
    def euler_from_quaternion(self, q):
        q0,q1,q2,q3 = q
        phi = np.arctan2(2*(q0*q1+q2*q3),1-2*(q1**2+q2**2))
        theta = np.arcsin(2*(q0*q2-q3*q1))
        psi = np.arctan((2*(q0*q3+q1*q2))/(1-2*(q2**2+q3**2)))
        return [phi,theta,psi]


    # Get current details for vehicle
    def odom_callback(self, msg):
        #x = -msg.pose.pose.position.y
        #y = msg.pose.pose.position.x
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        phi = self.euler_from_quaternion([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
                                                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[0]
        v_vec = msg.twist.twist.linear
        v = np.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)
        eps = 0.00001
        self.state = [0+eps,0+eps,0+eps,v+eps]
        self.state_hist.append(self.state)
        self.drive()


    def begin_drive(self):
        rate = self.create_rate(20)  # 10hz

        T = 5.0  # Time horizon
        N = 50  # number of control intervals

        # Declare model variables
        x1 = MX.sym('x1')
        x2 = MX.sym('x2')
        phi = MX.sym('phi')
        v = MX.sym('v')
        x = vertcat(x1, x2, phi, v)
        a = MX.sym('a')
        d = MX.sym('d')
        #d_d = MX.sym('dd')
        #d_a = MX.sym('da')
        u = vertcat(a, d)

        L_f = 0.76

        # Model equations
        xdot = vertcat(v * cos(phi), v * sin(phi), v * d / L_f, a)


        while rclpy.ok():

            t0 = time.time()

            #print("Drive")

            #print("State:")
            x_val,y_val,phi_val,v_val = self.state
            #print(self.state)

            #print("TARGET")
            #print(self.target_x)
            #print(self.target_y)

            # Objective term
            #L = (x1 - self.target_x) ** 2 + (x2 - self.target_y) ** 2 + 100*d**2 + 10*a**2
            c = np.sqrt(self.target_x**2+self.target_y**2)
            L = 1.0/(c**2)*(-x1*self.target_y + x2*self.target_x)**2 - 1.0/(10*c)*(x1*self.target_x + x2*self.target_y) + 25*d**2 + 5*a**2

            # Formulate discrete time dynamics
            # if False:
            # CVODES from the SUNDIALS suite
            #   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
            #   opts = {'tf':T/N}
            #   F = integrator('F', 'cvodes', dae, opts)

            DT = T / N
            f = Function('f', [x, u], [xdot, L])
            X0 = MX.sym('X0', 4)
            U = MX.sym('U', 2)
            X = X0
            Q = 0
            k1, k1_q = f(X, U)
            X = X + DT * k1
            Q = k1_q
            F = Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

            # Start with an empty NLP
            w = []
            w0 = []
            lbw = []
            ubw = []
            J = 0
            g = []
            lbg = []
            ubg = []

            # "Lift" initial conditions
            Xk = MX.sym('X0', 4)
            w += [Xk]
            lbw += [x_val, y_val, phi_val, v_val]
            ubw += [x_val, y_val, phi_val, v_val]
            w0 += [x_val, y_val, phi_val, v_val]

            # Formulate the NLP
            for k in range(N):
                # New NLP variable for the control
                Uk = MX.sym('U_' + str(k), 2)
                w += [Uk]
                lbw += [-1, -0.4]  # 25-degrees in radians
                ubw += [1, 0.4]
                w0 += [0, 0]

                # Integrate till the end of the interval
                Fk = F(x0=Xk, p=Uk)
                Xk_end = Fk['xf']
                J = J + Fk['qf']

                # New NLP variable for state at end of interval
                Xk = MX.sym('X_' + str(k + 1), 4)
                w += [Xk]
                lbw += [0, -inf, -1.6, 0]
                ubw += [inf, inf, 1.6, inf]
                w0 += [0, 0, 0, 0]

                # Add equality constraint
                g += [Xk_end - Xk]
                lbg += [0, 0, 0, 0]
                ubg += [0, 0, 0, 0]

            # Create an NLP solver
            prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
            opts = {'ipopt.print_level': 0, 'print_time': 0}
            solver = nlpsol('solver', 'ipopt', prob, opts)

            # Solve the NLP
            sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

            w_opt = sol['x'].full().flatten()

            # Plot the solution
            self.x1_opt = w_opt[0::6]
            self.x2_opt = w_opt[1::6]
            u1_opt = w_opt[4::6]
            u2_opt = w_opt[5::6]

            '''
            tgrid = [T/N*k for k in range(N+1)]
            import matplotlib.pyplot as plt
            plt.figure(1)
            plt.clf()
            plt.plot(tgrid, x1_opt, '--')
            plt.plot(tgrid, x2_opt, '-')
            plt.step(tgrid, vertcat(DM.nan(1), u1_opt), '-.')
            plt.step(tgrid, vertcat(DM.nan(1), u2_opt), '-.')
            plt.xlabel('t')
            plt.legend(['x1','x2','a','d'])
            plt.grid()
            plt.show()'''

            self.a.append(u1_opt[0])
            self.d.append(u2_opt[0])

            #print(u1_opt[0])
            #print(u2_opt[0])

            t1 = time.time()
            total = t1 - t0

            print("TIME DIFF")
            print(total)

            rate.sleep()

        self.destroy_rate(rate)

    def drive(self):
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()
        ack_msg.drive.steering_angle = self.d[-1]
        ack_msg.drive.acceleration = self.a[-1]
        self.ack_pub.publish(ack_msg)
        plt.clf()
        plt.plot(self.x1_opt, self.x2_opt)
        plt.savefig("prediction.png")
        plt.clf()
        plt.plot(self.d)
        plt.plot(self.a)
        plt.savefig("control-debug.png")
        plt.clf()
        plt.plot([s[0] for s in self.state_hist],[s[1] for s in self.state_hist])
        plt.savefig("trajectory.png")
        plt.clf()
        plt.plot([s[2] for s in self.state_hist], label="phi")
        plt.plot([s[3] for s in self.state_hist], label="v")
        plt.legend()
        plt.savefig("car-data.png")

    def keyboard_listener(self):
        # Collect events until released
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        #self.begin_drive()
        '''
        if key == Key.up:
            self.v += 0.1
        if key == Key.down:
            self.v += -0.1
        if key == Key.right:
            self.phi -= 0.1
        if key == Key.left:
            self.phi += 0.1'''


def main(args=None):
    rclpy.init(args=args)
    control = LocalController()
    thread = threading.Thread(target=rclpy.spin, args=(control,), daemon=True)
    thread.start()
    control.begin_drive()
    thread.join()
    rclpy.spin(control)
    rclpy.shutdown()




if __name__ == '__main__':
    main()