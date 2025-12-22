import carla
import math
import random

import os
import time
import argparse
import numpy as np

from config_set import Args
from network import SAC
from replaybuffer import ReplayBuffer

class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.desired_speed = 25
        self.stopping_distance = 15.0
        self.critical_distance = 5.0
        self.step_size = 10.0
        self.max_iterations = 100
        self.goal_sample_rate = 0.2
        self.min_distance_to_obstacle = 10
        self.search_radius = 100.0
        self.steer = 0
        self.avoidance_mode = None
        self.avoidance_timer = 0
        self.in_turn = False
        self.turn_timer = 0


    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        control = carla.VehicleControl()
        ego_x = transform.location.x
        ego_y = transform.location.y
        ego_yaw = transform.rotation.yaw
        current_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        yaw_rad = math.radians(ego_yaw)
        forward_x, forward_y = math.cos(yaw_rad), math.sin(yaw_rad)
        left_x, left_y = -forward_y, forward_x
        right_x, right_y = forward_y, -forward_x

        obstacle_left = False
        obstacle_right = False
        for obs in filtered_obstacles:
            ox, oy = obs.get_location().x, obs.get_location().y
            vec_x, vec_y = ox - ego_x, oy - ego_y
            dist = math.hypot(vec_x, vec_y)
            if dist < 20:
                proj_left = vec_x * left_x + vec_y * left_y
                proj_right = vec_x * right_x + vec_y * right_y
                if proj_left > 2.0:
                    obstacle_left = True
                if proj_right > 2.0:
                    obstacle_right = True

        target_x, target_y = self.get_target_waypoint(ego_x, ego_y, forward_x, forward_y, waypoints, boundary)

        if obstacle_left and not obstacle_right:
            self.avoidance_mode = "right"
            self.avoidance_timer = 20  # shorter duration
        elif obstacle_right and not obstacle_left:
            self.avoidance_mode = "left"
            self.avoidance_timer = 20
        elif not obstacle_left and not obstacle_right:
            #Reset avoidance mode immediately if no obstacles are near
            self.avoidance_mode = None
            self.avoidance_timer = 0
        elif self.avoidance_timer > 0:
            self.avoidance_timer -= 1
        else:
            self.avoidance_mode = None


        lateral_shift = 3
        if self.avoidance_timer > 0 and self.avoidance_mode is not None:
            decay = self.avoidance_timer / 20.0  # shorter decay window
            if self.avoidance_mode == "left":
                shift_x = lateral_shift * left_x
                shift_y = lateral_shift * left_y
            elif self.avoidance_mode == "right":
                shift_x = lateral_shift * right_x
                shift_y = lateral_shift * right_y
            target_x += shift_x * decay
            target_y += shift_y * decay

        path = self.rrt_plan(ego_x, ego_y, target_x, target_y, filtered_obstacles, boundary)
        next_x, next_y = path[10] if path and len(path) > 10 else (target_x, target_y)

        max_action = np.array(Args['max_action'])
        max_all_step = Args['max_all_step']
        agent = SAC(max_action,
                    Args['a_lr'],
                    Args['c_lr'],
                    Args['sac_gamma'],
                    Args['sac_tau'],
                    Args['sac_alpha'],
                    Args['sac_policy'],
                    Args['sac_policy_freq'],
                    Args['sac_auto_entropy_tuning'],
                    Args['debug'],
                    Args['back_enable'])
        if Args['model_load']:
            agent.load(Args['train_file'], Args['model_path'])
        if Args['model_save'] or Args['check_save']:
            now = time.localtime()
            save_file = ''.join(
                ('SAC_model_', str(now.tm_year), str(now.tm_mon), str(now.tm_mday), str(now.tm_hour), str(now.tm_min),
                 str(now.tm_sec)))
            save_path = os.path.join(Args['model_path'], save_file)
            try:
                os.mkdir(save_path)
            except:
                print('Failed to create model folder.')

        replay_buffer = ReplayBuffer(Args['frame_size'],
                                     Args['state_size'],
                                     Args['action_size'],
                                     Args['buffer_size'],
                                     Args['buffer_seed'])

        done = True  ## episode done
        restart = False  ## episode discard & restart
        episode = -1  ## current episode
        total_step = 0  ## total step
        done_count = 0  ## count of success episode
        loss_set = []  ## loss record
        episode_set = []  ## performance record
        train_time = time.time()  ## total used time



        control.throttle = throttle
        control.brake = brake
        return control

    def get_target_waypoint(self, ego_x, ego_y, fx, fy, waypoints, boundary):
        for wp in waypoints:
            wp_x, wp_y = wp[0], wp[1]
            vec_x, vec_y = wp_x - ego_x, wp_y - ego_y
            if vec_x * fx + vec_y * fy > 0:
                return wp_x, wp_y
        return self.get_centerline_target(ego_x, ego_y, 40, boundary)

    def get_centerline_target(self, ego_x, ego_y, lookahead, boundary):
        left, right = boundary
        min_dist, best_pt = float('inf'), None
        for l, r in zip(left, right):
            cx = (l.transform.location.x + r.transform.location.x) / 2.0
            cy = (l.transform.location.y + r.transform.location.y) / 2.0
            dist = math.hypot(cx - ego_x, cy - ego_y)
            if dist >= lookahead and dist < min_dist:
                best_pt = (cx, cy)
                min_dist = dist
        return best_pt or ((left[-1].transform.location.x + right[-1].transform.location.x) / 2.0,
                           (left[-1].transform.location.y + right[-1].transform.location.y) / 2.0)

    def rrt_plan(self, sx, sy, gx, gy, obstacles, boundary):
        tree = [(sx, sy, -1)]
        left, right = boundary
        for _ in range(self.max_iterations):
            rx, ry = (gx, gy) if random.random() < self.goal_sample_rate else (
                sx + (random.random() - 0.5) * 2 * self.search_radius,
                sy + (random.random() - 0.5) * 2 * self.search_radius)
            idx, nx, ny = self.find_nearest(tree, rx, ry)
            theta = math.atan2(ry - ny, rx - nx)
            new_x = nx + self.step_size * math.cos(theta)
            new_y = ny + self.step_size * math.sin(theta)
            if self.is_collision_free(nx, ny, new_x, new_y, obstacles, left, right):
                tree.append((new_x, new_y, idx))
                if math.hypot(new_x - gx, new_y - gy) < self.step_size:
                    return self.reconstruct_path(tree, len(tree) - 1) + [(gx, gy)]
        idx, _, _ = self.find_nearest(tree, gx, gy)
        return self.reconstruct_path(tree, idx)

    def find_nearest(self, tree, x, y):
        min_dist, idx = float('inf'), 0
        for i, (tx, ty, _) in enumerate(tree):
            dist = math.hypot(tx - x, ty - y)
            if dist < min_dist:
                idx = i
                min_dist = dist
        return idx, tree[idx][0], tree[idx][1]

    def reconstruct_path(self, tree, idx):
        path = []
        while idx != -1:
            x, y, idx = tree[idx]
            path.append((x, y))
        return path[::-1]

    def is_collision_free(self, x1, y1, x2, y2, obstacles, left, right):
        for obs in obstacles:
            ox, oy = obs.get_location().x, obs.get_location().y
            if self.point_to_segment_distance(ox, oy, x1, y1, x2, y2) < self.min_distance_to_obstacle * 1.5:
                return False
        for bounds in (left, right):
            for i in range(len(bounds) - 1):
                a = bounds[i].transform.location
                b = bounds[i + 1].transform.location
                if self.line_intersection(x1, y1, x2, y2, a.x, a.y, b.x, b.y)[0]:
                    return False
        return True

    def point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return False, float('inf')
        t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4)) / denom
        u = -((x1 - x2)*(y1 - y3) - (y1 - y2)*(x1 - x3)) / denom
        return (0 <= t <= 1 and 0 <= u <= 1), 0
