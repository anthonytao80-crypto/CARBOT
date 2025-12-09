import carla
import math
import random
import numpy as np
from collections import deque
import sys
import glob
import os
class Agent():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.desired_speed = 45
        self.stopping_distance = 15.0
        self.critical_distance = 5.0
        self.step_size = 10.0
        self.max_iterations = 50
        self.goal_sample_rate = 0.2
        self.min_distance_to_obstacle = 10
        self.search_radius = 100.0
        self.avoidance_mode = None
        self.avoidance_timer = 0

        self._lon_controller = PIDLongitudinalController(self.vehicle, 15, 0.01, 0)
        self._lat_controller = PIDLateralController(self.vehicle, 0, 2.1, 0.01, 0)

    # ======================= Run Step =======================
    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        control = carla.VehicleControl()
        ego_x, ego_y = transform.location.x, transform.location.y
        ego_yaw = transform.rotation.yaw
        ego_yaw_rad = math.radians(ego_yaw)
        current_speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        fx, fy = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
        lx, ly = -fy, fx  # 左方向向量

        # =================== 1) 目标点 ===================
        target_x, target_y, need_brake = self.compute_dynamic_target(
            ego_x, ego_y, ego_yaw, filtered_obstacles, boundary, lookahead=8.0
        )

        # =================== 2) RRT 路径规划 ===================
        left, right = boundary
        path = self.rrt_plan(ego_x, ego_y, target_x, target_y, filtered_obstacles, boundary)
        next_x, next_y = path[10] if path and len(path) > 10 else (target_x, target_y)

        # =================== 3) 横向控制 ===================
        control.steer = self._lat_controller.run_step(ego_x, ego_y, next_x, next_y, ego_yaw)

        # =================== 4) 纵向控制 ===================
        v_desired = self.calculate_speed_control(
            ego_x, ego_y, ego_yaw, filtered_obstacles, current_speed,
            control.steer, left, right, (target_x, target_y), waypoints
        )
        acceleration = self._lon_controller.run_step(current_speed, v_desired)

        if acceleration >= 0.0:
            control.throttle = min(acceleration, 0.75)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), 0.5)

        # =================== 5) 紧急制动 ===================
        if need_brake or self.Emergency_brake(ego_x, ego_y, ego_yaw, filtered_obstacles, current_speed, left, right):
            control.brake = 1.0
            control.throttle = 0.0

        return control

    # ======================= 紧急制动 =======================
    def Emergency_brake(self, ego_x, ego_y, ego_yaw, obstacles, speed, left, right):
        ego_yaw_rad = math.radians(ego_yaw)
        fx, fy = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
        self.stopping_distance = speed ** 2 / 10

        # 只考虑前方 ±10°，距离 < stopping_distance + margin 的障碍物
        for obs in obstacles:
            ox, oy = obs.get_location().x, obs.get_location().y
            dx, dy = ox - ego_x, oy - ego_y
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            dot = (dx * fx + dy * fy) / dist
            if dot > math.cos(math.radians(10)) and dist < self.stopping_distance + 5:
                return True
        return False


    # ======================= 动态目标点 =======================
    def compute_dynamic_target(self, ego_x, ego_y, ego_yaw, filtered_obstacles, boundary, lookahead=20.0, min_gap=7.0, lateral_shift=3.0):
        yaw_rad = math.radians(ego_yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        lx, ly = -fy, fx

        left_pts, right_pts = boundary
        center_x, center_y = self.get_centerline_target(ego_x, ego_y, lookahead, boundary)

        # 车道左右边界横向坐标
        lpt = left_pts[0].transform.location
        rpt = right_pts[0].transform.location
        lane_min = min((lpt.x - center_x)*lx + (lpt.y - center_y)*ly,
                       (rpt.x - center_x)*lx + (rpt.y - center_y)*ly)
        lane_max = max((lpt.x - center_x)*lx + (lpt.y - center_y)*ly,
                       (rpt.x - center_x)*lx + (rpt.y - center_y)*ly)

        # 障碍物横向位置
        obs_lat = np.array([((obs.get_location().x - ego_x)*lx + (obs.get_location().y - ego_y)*ly)
                            for obs in filtered_obstacles if math.hypot(obs.get_location().x - ego_x, obs.get_location().y - ego_y) < 20])

        if len(obs_lat) == 0:
            return center_x, center_y, False

        # 找最大横向空隙
        positions = np.concatenate(([lane_min], np.sort(obs_lat), [lane_max]))
        gaps = positions[1:] - positions[:-1]
        max_idx = np.argmax(gaps)
        max_gap = gaps[max_idx]
        gap_left, gap_right = positions[max_idx], positions[max_idx+1]

        if max_gap > min_gap:
            best_lat = (gap_left + gap_right)/2
            target_x = center_x + best_lat * lx
            target_y = center_y + best_lat * ly
            need_brake = False
        else:
            target_x, target_y = center_x, center_y
            need_brake = True

        # 避障偏置衰减
        if self.avoidance_timer > 0 and self.avoidance_mode:
            decay = self.avoidance_timer / 20.0
            sx, sy = (lx, ly) if self.avoidance_mode == "left" else (-lx, -ly)
            target_x += lateral_shift * decay * sx
            target_y += lateral_shift * decay * sy

        return target_x, target_y, need_brake

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

    def calculate_steering(self, ego_x, ego_y, ego_yaw, target_x, target_y):
        yaw_rad = math.radians(ego_yaw)
        dx = target_x - ego_x
        dy = target_y - ego_y
        angle_diff = math.atan2(dy, dx) - yaw_rad
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        return max(min(angle_diff / (math.pi / 2), 1.0), -1.0)


    def is_sharp_turn_ahead(self, ego_x, ego_y, waypoints, lookahead=5):
        if len(waypoints) < lookahead + 1:
            return False

        # Vector from current pos to next point
        dx1 = waypoints[1][0] - ego_x
        dy1 = waypoints[1][1] - ego_y
        dx2 = waypoints[lookahead][0] - waypoints[1][0]
        dy2 = waypoints[lookahead][1] - waypoints[1][1]

        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        angle_diff = abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)

        return angle_diff > math.radians(30)  # tweak this threshold


    def estimate_road_width(self, left, right, index=5):
        if len(left) <= index or len(right) <= index:
            return float('inf')  # no boundary info
        lx, ly = left[index].transform.location.x, left[index].transform.location.y
        rx, ry = right[index].transform.location.x, right[index].transform.location.y
        return math.hypot(rx - lx, ry - ly)

    def calculate_speed_control(self, ego_x, ego_y, ego_yaw, obstacles, speed, steer, left, right, target, waypoints):

        yaw_rad = math.radians(ego_yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)

        # 默认期望速度
        v_desired = self.desired_speed

        # 曲率估计
        curve_radius = self.estimate_curvature_radius(waypoints)

        # ------------------------------
        # 弯道减速逻辑（原版）
        # ------------------------------
        steer_strength = abs(steer)

        # 小曲率 → 弯道限速
        if curve_radius < 30:
            v_desired = min(v_desired, 10.0)

        # 中等转角 → 平滑限速
        if steer_strength > 0.3:
            turn_speed = max(15, self.desired_speed * (1.0 - steer_strength))
            v_desired = min(v_desired, turn_speed)

        return v_desired

    def estimate_curvature_radius(self, waypoints, lookahead=5):
        if len(waypoints) < lookahead + 1:
            return float('inf')

        # Get 3 spaced points
        p1 = waypoints[1]
        p2 = waypoints[lookahead // 2]
        p3 = waypoints[lookahead]

        # Compute circle through three points
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]

        A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
        C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
        D = (x1**2 + y1**2) * (x3 * y2 - x2 * y3) + (x2**2 + y2**2) * (x1 * y3 - x3 * y1) + (x3**2 + y3**2) * (x2 * y1 - x1 * y2)

        if A == 0:
            return float('inf')  # points are colinear

        cx = -B / (2 * A)
        cy = -C / (2 * A)
        radius = math.sqrt((cx - x1)**2 + (cy - y1)**2)

        return radius


    def distance_to_edge(self, ego_x, ego_y, left, right, index=5):
        if index >= len(left) or index >= len(right):
            return float('inf')
        lx, ly = left[index].transform.location.x, left[index].transform.location.y
        rx, ry = right[index].transform.location.x, right[index].transform.location.y
        center_x = (lx + rx) / 2
        center_y = (ly + ry) / 2
        dist = math.hypot(ego_x - center_x, ego_y - center_y)
        return dist

class PIDLongitudinalController():
    """
    Implements longitudinal vehicle control using a PID controller.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the PIDLongitudinalController.

        Args:
            vehicle: The CARLA vehicle actor.
            K_P (float): Proportional gain.
            K_D (float): Differential gain.
            K_I (float): Integral gain.
            dt (float): Time step (in seconds) for integration and differentiation.
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self,current_speed, target_speed, debug=False):
        """
        Compute the PID control action to drive the vehicle toward the target speed.

        Args:
            target_speed (float): The desired speed in Km/h.
            debug (bool): If True, prints debug information.

        Returns:
            float: The computed control value (positive for throttle, negative for brake).
        """
        if debug:
            print('Current speed = {}'.format(current_speed))
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Perform the internal PID calculation for longitudinal control.

        Args:
            target_speed (float): The desired speed in Km/h.
            current_speed (float): The current speed in Km/h.

        Returns:
            float: The PID controller output (clipped between -1 and 1).
        """
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)


class PIDLateralController():
    """
    Implements lateral vehicle control using a PID controller.
    """

    def __init__(self, vehicle, offset=0, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        Initialize the PIDLateralController.

        Args:
            vehicle: The CARLA vehicle actor.
            offset (float): Lateral offset from the center line. A nonzero value displaces the target waypoint.
            K_P (float): Proportional gain.
            K_D (float): Differential gain.
            K_I (float): Integral gain.
            dt (float): Time step (in seconds) for integration and differentiation.
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, ego_x,ego_y,next_x,next_y,ego_yaw):
        """
        Compute the lateral control action to steer the vehicle toward the target waypoint.

        Args:
            waypoint: The target waypoint (with transform information).

        Returns:
            float: Steering control value in the range [-1, 1] where -1 indicates maximum left steering and +1 indicates maximum right steering.
        """
        return self._pid_control(ego_x,ego_y,next_x,next_y,ego_yaw)

    def _pid_control(self, ego_x,ego_y,next_x,next_y,ego_yaw):
        """
        Perform the internal PID calculation for lateral control.

        This function computes the angular error between the vehicle's forward vector and the vector pointing from the vehicle
        to the target waypoint. It then applies a PID correction to compute the steering command.

        Args:
            waypoint: The target waypoint.
            vehicle_transform: The current transform of the vehicle.

        Returns:
            float: The computed steering command (clipped between -1 and 1).
        """
        # Get the vehicle's current position and forward vector.
        # Create the vector from the vehicle to the waypoint.
        w_vec = np.array([next_x - ego_x,
                          next_y - ego_y,
                          0.0])
        ego_yaw = math.radians(ego_yaw)
        v_vec = np.array([
            math.cos(ego_yaw),
            math.sin(ego_yaw),
            0.0
        ])
        # Compute the angular error between the vehicle's heading and the direction to the waypoint.
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))
        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        # Update the error buffer and compute derivative and integral terms.
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * _dot) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)