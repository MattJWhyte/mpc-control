#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports
# Python
from pynput.keyboard import Listener, Key, Controller

# ROS
import rclpy
from rclpy.node import Node
from eufs_msgs.msg import ConeArrayWithCovariance
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
import threading
import shutil
import os
from scipy.optimize import minimize

class TelemetryNode(Node):

    def __init__(self):
        super().__init__("telemetry_node")
        self.cone_sub = self.create_subscription(
        ConeArrayWithCovariance,
        '/fusion/cones',
        self.cone_callback,
        5)
        self.pose_sub = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            5)
        self.drawing = False
        self.count = 0
        self.odom_ct = 0
        self.path = [[],[]]
        shutil.rmtree('track')
        os.mkdir("track")

    def cone_callback(self, msg):
        if not self.drawing:
            self.drawing = True
        else:
            return
        Bx = []
        By = []
        Yx = []
        Yy = []
        for c in msg.blue_cones:
            Bx.append(-c.point.y)
            By.append(c.point.x)
        for c in msg.yellow_cones:
            Yx.append(-c.point.y)
            Yy.append(c.point.x)

        plt.clf()

        B_lane = None
        bs_y,bs_x = None, None

        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        n = len(Bx)
        if n > 1:
            Bx, By = self.sort_cones(Bx, By)
            t = np.linspace(0,n-1,n)
            bs_x = CubicSpline(t, np.array(Bx))
            bs_y = CubicSpline(t, np.array(By))

            sample_t = np.linspace(0,n-1,100)
            sample_x = bs_x(sample_t)
            sample_y = bs_y(sample_t)
            plt.plot(sample_x,sample_y, 'b')
            B_lane = sample_x, sample_y

        Y_lane = None
        ys_y, ys_x = None, None

        n = len(Yx)
        if n > 1:
            Yx, Yy = self.sort_cones(Yx, Yy)
            t = np.linspace(0, n - 1, n)
            ys_x = CubicSpline(t, np.array(Yx))
            ys_y = CubicSpline(t, np.array(Yy))

            sample_t = np.linspace(0, n-1, 100)
            sample_x = ys_x(sample_t)
            sample_y = ys_y(sample_t)
            plt.plot(sample_x, sample_y, 'y')
            Y_lane = sample_x, sample_y

        if Y_lane is not None and B_lane is not None:

            data = np.array([Bx,By,Yx,Yy])
            np.save("track/data.npy",data)
            mid_x = (Y_lane[0]+B_lane[0])/2.0
            mid_y = (Y_lane[1]+B_lane[1])/ 2.0
            #plt.plot(mid_x, mid_y, 'r')

            X,Y = np.meshgrid(np.linspace(-7.5,7.5,30),np.linspace(0,15,30))

            '''
            D = np.zeros(X.shape)
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    D[i,j] = self.distance(bs_x, bs_y, ys_x, ys_y, X[i,j], Y[i,j])
                    '''

            #ax = fig.gca(projection='3d')
            #ax.plot_surface(X,Y,D)

        plt.scatter(Bx,By,c='b')
        plt.scatter(Yx,Yy, c='y')
        plt.savefig("track/{}.png".format(self.count))
        self.count += 1
        self.drawing = False


    def closest(self, x, y, xx, yy):
        D = np.stack((np.array(xx) - x, np.array(yy) - y), axis=1)
        D = np.sum(D ** 2, axis=1)
        return np.argmin(D)


    def distance(self, bs_x, bs_y, ys_x, ys_y, x, y):
        d = lambda t : (0.5*(bs_x(t)+ys_x(t)) - x)**2+(0.5*(bs_y(t)+ys_y(t)) - y)**2
        return minimize(d, 0).x[0]


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


    def odom_callback(self, msg):
        self.odom_ct += 1
        self.path[0].append(-msg.pose.pose.position.y)
        self.path[1].append(msg.pose.pose.position.x)
        if self.odom_ct % 5 != 0:
            return
        if not self.drawing:
            self.drawing = True
        else:
            return
        plt.clf()
        plt.plot(self.path[0], self.path[1])
        plt.savefig("path.png")
        self.drawing = False

def main(args=None):
    rclpy.init(args=args)
    telemetry = TelemetryNode()
    rclpy.spin(telemetry)
    rclpy.shutdown()


if __name__ == '__main__':
    main()


'''
def fit_nn_gradopt(X, yy, alpha, init):
    D = X.shape[1]
    args = (X, yy, alpha)
    ww, bb, V, bk = minimize_list(nn_cost, init, args)
    #nn_cost(params, X, yy=None, alpha=None)
    return ww, bb, V, bk

def rand_init(shape, C):
    return 0.1*np.random.randn(*shape)/np.sqrt(C)
    
ww_rand_init = rand_init([20], 20)
bb_rand_init = rand_init([], 1)
V_rand_init = rand_init([20,373], 373)
bk_rand_init = rand_init([20], 1)

ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init = fit_nn_gradopt(X_train, y_train, 30, [ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init])
X_train_pred = nn_cost([ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init], X_train)
X_val_pred = nn_cost([ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init], X_val)

print("Random Initialisation")
rmse_train = rmse(y_train, X_train_pred)
print("RMSE on X_train : {}".format(rmse_train))

rmse_val = rmse(y_val, X_val_pred)
print("RMSE on X_val : {}".format(rmse_val))

ww_q3_init,bb_q3_init,V_q3_init,bk_q3_init = fit_nn_gradopt(X_train, y_train, 30, [ww_linreg[:,0], bb_linreg[0], V, bk])
X_train_pred = nn_cost([ww_q3_init,bb_q3_init,V_q3_init,bk_q3_init], X_train)
X_val_pred = nn_cost([ww_q3_init,bb_q3_init,V_q3_init,bk_q3_init], X_val)

print("\nQ3 Params Initialisation")
rmse_train = rmse(y_train, X_train_pred)
print("RMSE on X_train : {}".format(rmse_train))

rmse_val = rmse(y_val, X_val_pred)
print("RMSE on X_val : {}".format(rmse_val))

def train_nn_reg(alpha):
    ww_rand_init = rand_init([20], 20)
    bb_rand_init = rand_init([], 1)
    V_rand_init = rand_init([20,373], 373)
    bk_rand_init = rand_init([20], 1)

    ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init = fit_nn_gradopt(X_train, y_train, alpha, [ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init])
    X_val_pred = nn_cost([ww_rand_init,bb_rand_init,V_rand_init,bk_rand_init], X_val)
    return rmse(y_val, X_val_pred)

print(train_nn_reg(30))
'''