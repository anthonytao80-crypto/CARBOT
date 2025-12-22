#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Yan's Note: GRAIC 2023 -- NO ROS VERSION"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
from turtle import right
import numpy.random as random
import re
import sys
import weakref

import pickle
import time
import subprocess

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agent import Agent
from network import SAC
from replaybuffer import ReplayBuffer
from config_set import Args, state_size, frame_size, action_size, max_action
import torch


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def extract_state(vehicle, waypoints, idx, filtered_obstacles, boundary, vel, transform, prev_action=None):
    """
    提取状态向量
    state_size = goal_size(2) + action_size(3) + current_state_size(3) + 
                 boundary_size(4) + filtered_obstacles_size(20) + current_speed_size(1)
    """
    # 目标点 (goal_size = 2)
    if idx < len(waypoints):
        target_x, target_y = waypoints[idx][0], waypoints[idx][1]
    else:
        target_x, target_y = waypoints[-1][0], waypoints[-1][1]
    
    # 当前状态 (current_state_size = 3)
    cur_pos = vehicle.get_location()
    cur_x, cur_y = cur_pos.x, cur_pos.y
    cur_yaw = transform.rotation.yaw
    
    # 边界信息 (boundary_size = 4)
    # 获取最近的左右边界点
    left_boundary, right_boundary = boundary
    if left_boundary and right_boundary:
        # 使用最近的边界点
        left_wp = left_boundary[0] if left_boundary else None
        right_wp = right_boundary[0] if right_boundary else None
        if left_wp and right_wp:
            lx = left_wp.transform.location.x
            ly = left_wp.transform.location.y
            rx = right_wp.transform.location.x
            ry = right_wp.transform.location.y
        else:
            lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0
    else:
        lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0
    
    # 障碍物信息 (filtered_obstacles_size = 4*5 = 20)
    # 最多5个障碍物，每个4个值 (ox, oy, vx, vy)
    obstacle_features = np.zeros(20)
    for i, obs in enumerate(filtered_obstacles[:5]):
        obs_loc = obs.get_location()
        obs_vel = obs.get_velocity()
        obstacle_features[i*4] = obs_loc.x
        obstacle_features[i*4+1] = obs_loc.y
        obstacle_features[i*4+2] = obs_vel.x
        obstacle_features[i*4+3] = obs_vel.y
    
    # 速度 (current_speed_size = 1)
    speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
    # 上一动作 (action_size = 3)
    if prev_action is None:
        prev_steer, prev_throttle, prev_brake = 0.0, 0.0, 0.0
    else:
        # prev_action可能是numpy数组或VehicleControl对象
        if isinstance(prev_action, np.ndarray):
            prev_steer = prev_action[0] if len(prev_action) > 0 else 0.0
            prev_throttle = prev_action[1] if len(prev_action) > 1 else 0.0
            prev_brake = prev_action[2] if len(prev_action) > 2 else 0.0
        else:
            prev_steer = prev_action.steer
            prev_throttle = prev_action.throttle
            prev_brake = prev_action.brake
    
    # 组合状态向量
    state = np.array([
        target_x, target_y,  # goal (2)
        prev_steer, prev_throttle, prev_brake,  # action (3)
        cur_x, cur_y, cur_yaw,  # current state (3)
        lx, ly, rx, ry,  # boundary (4)
        *obstacle_features,  # obstacles (20)
        speed  # speed (1)
    ], dtype=np.float32)
    
    return state


def calculate_reward(vehicle, waypoints, idx, distance, collision_occurred, reached_goal, 
                     prev_distance=None, speed=None):
    """
    计算奖励
    """
    reward = 0.0
    
    # 到达终点奖励
    if reached_goal:
        reward += Args.get('arrived_R', 60.0)
    
    # 碰撞惩罚
    if collision_occurred:
        reward += Args.get('collision_R', -80.0)
        return reward
    
    # 距离奖励：越接近目标点奖励越高
    if prev_distance is not None:
        distance_improvement = prev_distance - distance
        reward += distance_improvement * 0.1
    
    # 速度奖励：保持合理速度
    if speed is not None:
        desired_speed = 25.0  # km/h
        speed_kmh = speed * 3.6
        if 15.0 < speed_kmh < 35.0:
            reward += 0.1
        elif speed_kmh < 5.0:
            reward -= 0.5  # 太慢惩罚
    
    # 存活奖励
    reward += 0.01
    
    return reward


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'ego_vehicle')
        if blueprint.has_attribute('color'):
            # color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', "0,255,0")

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            # TODO: Hardcoded starting point, to be changed to 1st wp in waypoint pickle file
            if args.map == 'shanghai_intl_circuit':
                spawn_point = carla.Transform(carla.Location(90.0, 93.0, 1.0), carla.Rotation(0, 0, 0))
            elif args.map == 't1_triple':
                spawn_point = carla.Transform(carla.Location(153.0, -12.0, 1.0), carla.Rotation(0, 180, 0))
            elif args.map == 't2_triple':
                spawn_point = carla.Transform(carla.Location(85.0, -105.0, 1.0), carla.Rotation(0, 135, 0))
            elif args.map == 't3':
                spawn_point = carla.Transform(carla.Location(70.0, -115.0, 1.0), carla.Rotation(0, 16.75, 0))
            elif args.map == 't4':
                spawn_point = carla.Transform(carla.Location(155.0, -96.0, 1.0), carla.Rotation(0, 50, 0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud, args)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Welcome to GRAIC Competition", seconds=2.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud, args):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

        self.prev_collision_actor_id = []
        self.args = args

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        
        if event.other_actor.id not in self.prev_collision_actor_id:
            self.prev_collision_actor_id.append(event.other_actor.id)
            with open("{}_collision.txt".format(self.args.map), 'a') as f:
                f.write("Time: {}, Collision Actor ID: {}, Collision Actor Type: {}\n".format(round(event.timestamp, 2), event.other_actor.id, event.other_actor.type_id))

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        # self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation with SAC training.
    It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.load_world(args.map)

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(args.seed)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # 初始化SAC网络和ReplayBuffer
        sac_agent = SAC(
            max_action=np.array(Args['max_action']),
            actor_lr=Args['a_lr'],
            critic_lr=Args['c_lr'],
            gamma=Args['sac_gamma'],
            tau=Args['sac_tau'],
            alpha=Args['sac_alpha'],
            policy=Args['sac_policy'],
            policy_freq=Args['sac_policy_freq'],
            auto_tuning=Args['sac_auto_entropy_tuning'],
            debug=Args['debug'],
            enable_backward=Args['back_enable']
        )

        # 加载模型（如果存在）
        if Args['model_load']:
            try:
                sac_agent.load(Args['train_file'], Args['model_path'])
                print("Model loaded successfully")
            except Exception as e:
                print(f"Failed to load model: {e}")

        replay_buffer = ReplayBuffer(
            s1_frame=Args['frame_size'],
            s1_dim=Args['state_size'],
            a_dim=Args['action_size'],
            buffer_size=Args['buffer_size'],
            random_seed=Args['buffer_seed']
        )

        # 加载waypoints
        original_wps = pickle.load(open("./waypoints/{}".format(args.map), 'rb'))
        waypoints = original_wps[1:]

        clock = pygame.time.Clock()
        
        # 训练相关变量
        episode = 0
        total_steps = 0
        state_history = []  # 用于存储frame_size个连续状态
        prev_action = None
        prev_distance = None
        episode_reward = 0.0
        episode_steps = 0
        loss_history = []

        # 训练循环
        while True:
            # Episode开始：重置环境
            idx = 0
            episode += 1
            episode_reward = 0.0
            episode_steps = 0
            state_history = []
            prev_action = None
            prev_distance = None
            prev_reward = 0.0
            prev_state_input = None
            collision_occurred = False
            reached_goal = False

            # 重置碰撞传感器历史
            world.collision_sensor.prev_collision_actor_id = []
            
            print(f"\n=== Episode {episode} ===")

            # 主循环：单个episode
            while True:
                clock.tick()
                if args.sync:
                    world.world.tick()
                else:
                    world.world.wait_for_tick()
                
                if controller.parse_events():
                    return
                
                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                vehicle = world.player
                cur_pos = vehicle.get_location()
                vel = vehicle.get_velocity()
                transform = vehicle.get_transform()

                cur_x, cur_y, cur_z = cur_pos.x, cur_pos.y, cur_pos.z
                
                # 检查是否到达终点
                if idx < len(waypoints):
                    target_x, target_y, target_z = waypoints[idx]
                    distance = math.sqrt((cur_x - target_x)**2 + (cur_y-target_y)**2)
                    
                    if distance < 5:
                        idx += 1
                        if idx == len(waypoints):
                            reached_goal = True
                            print(f"Episode {episode}: Reached goal!")
                            break
                else:
                    reached_goal = True
                    print(f"Episode {episode}: Reached goal!")
                    break

                # 检查碰撞
                collision_history = world.collision_sensor.get_collision_history()
                recent_collisions = [collision_history.get(world.hud.frame - i, 0) for i in range(10)]
                if any(c > 0 for c in recent_collisions):
                    collision_occurred = True
                    print(f"Episode {episode}: Collision detected!")
                    break

                # 获取障碍物
                all_actors = world.world.get_actors()
                sensoring_radius = 100
                filtered_obstacles = []
                for actor in all_actors:
                    cur_loc = actor.get_location()
                    if cur_pos.distance(cur_loc) <= sensoring_radius:
                        if 'vehicle' in actor.type_id and actor.id != vehicle.id:
                            filtered_obstacles.append(actor)
                        elif 'pedestrian' in actor.type_id:
                            filtered_obstacles.append(actor)
                        elif 'static.prop' in actor.type_id:
                            filtered_obstacles.append(actor)

                # 获取边界信息
                cur_waypoint = world.map.get_waypoint(cur_pos)
                cur_left = cur_waypoint
                cur_right = cur_waypoint
                left_boundary, right_boundary = [], []

                # Get Left Boundary
                while cur_left.get_left_lane():
                    cur_left = cur_left.get_left_lane()
                for i in range(50):
                    left_boundary.extend(cur_left.next(i+0.5))

                # Get Right Boundary
                while cur_right.get_right_lane():
                    cur_right = cur_right.get_right_lane()
                for i in range(50):
                    right_boundary.extend(cur_right.next(i+0.5))

                boundary = [left_boundary, right_boundary]

                # 提取当前状态
                current_state = extract_state(
                    vehicle, waypoints, idx, filtered_obstacles, 
                    boundary, vel, transform, prev_action
                )

                # 更新状态历史
                state_history.append(current_state)
                if len(state_history) > frame_size:
                    state_history.pop(0)

                # 确保有足够的状态历史
                if len(state_history) < frame_size:
                    # 用当前状态填充历史
                    while len(state_history) < frame_size:
                        state_history.insert(0, current_state.copy())
                
                # 准备LSTM输入（需要frame_size个状态）
                state_input = np.array(state_history[-frame_size:]).reshape(1, frame_size, state_size)

                # 获取动作
                if total_steps < Args['warm_up']:
                    # Warm-up阶段：随机动作
                    action = np.random.uniform(
                        low=-1.0, high=1.0, size=action_size
                    )
                else:
                    # 使用SAC网络获取动作
                    action = sac_agent.get_action(state_input, is_train=True)

                # 将动作转换为控制命令
                control = carla.VehicleControl()
                control.steer = float(np.clip(action[0], -1.0, 1.0))
                control.throttle = float(np.clip(action[1], 0.0, 1.0))
                control.brake = float(np.clip(action[2], 0.0, 1.0))
                control.manual_gear_shift = False
                world.player.apply_control(control)

                # 计算奖励
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                reward = calculate_reward(
                    vehicle, waypoints, idx, distance, 
                    collision_occurred, reached_goal,
                    prev_distance, speed
                )
                episode_reward += reward

                # 存储经验（需要上一个状态和动作）
                if prev_action is not None and prev_state_input is not None and len(state_history) >= frame_size:
                    # 使用当前状态作为下一个状态（因为这是当前step的状态）
                    # 存储到replay buffer
                    done_flag = 1 if (collision_occurred or reached_goal) else 0
                    replay_buffer.add(
                        prev_state_input[0],  # 上一个状态序列
                        prev_action,  # 上一个动作
                        prev_reward,  # 上一个奖励
                        state_input[0],  # 当前状态序列（作为下一个状态）
                        done_flag  # 是否结束
                    )

                # 更新变量（保存当前状态用于下一个step）
                prev_action = action.copy() if isinstance(action, np.ndarray) else action
                prev_distance = distance
                prev_reward = reward
                prev_state_input = state_input.copy()
                episode_steps += 1
                total_steps += 1

                # 训练网络
                if total_steps > Args['warm_up'] and replay_buffer.get_capacity() > Args['batch_size']:
                    loss_set = sac_agent.train(
                        replay_buffer,
                        iteration=1,
                        batch_size=Args['batch_size'],
                        use_replay=Args['use_replay']
                    )
                    if loss_set:
                        loss_history.extend(loss_set)

                # 如果episode结束，存储最后一个经验并跳出内层循环
                if collision_occurred or reached_goal:
                    # 存储最后一个经验
                    if prev_action is not None and len(state_history) >= frame_size:
                        replay_buffer.add(
                            state_input[0],  # 当前状态序列
                            action,  # 当前动作
                            reward,  # 当前奖励
                            state_input[0],  # 下一个状态（episode结束，使用相同状态）
                            1  # done = 1
                        )
                    break

            # Episode结束，打印统计信息
            print(f"Episode {episode} finished:")
            print(f"  Steps: {episode_steps}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Collision: {collision_occurred}, Reached Goal: {reached_goal}")

            # 保存模型（定期保存）
            if Args['model_save'] and episode % Args.get('check_step', 100) == 0:
                try:
                    import os
                    import time
                    now = time.localtime()
                    save_file = ''.join((
                        'SAC_model_', str(now.tm_year), str(now.tm_mon), str(now.tm_mday),
                        str(now.tm_hour), str(now.tm_min), str(now.tm_sec)
                    ))
                    save_path = os.path.join(Args['model_path'], save_file)
                    try:
                        os.makedirs(save_path, exist_ok=True)
                    except:
                        print('Failed to create model folder.')
                    sac_agent.save(save_file, Args['model_path'])
                    print(f"Model saved to {save_path}")
                except Exception as e:
                    print(f"Failed to save model: {e}")

            # Restart world for next episode
            world.restart(args)
            time.sleep(0.1)  # 短暂延迟确保重启完成

    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        # TODO: change for other vehicle models
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=1234,
        type=int)
    argparser.add_argument(
        '-m', '--map',
        help='Set Different Map for testing: shanghai_intl_circuit, t1_triple, t2_triple, t3, t4',
        default="shanghai_intl_circuit")

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
