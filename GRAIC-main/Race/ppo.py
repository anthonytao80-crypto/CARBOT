# carla_gym_env.py
import gym
import numpy as np
import carla
import random
import time
from gym import spaces
import cv2

class CarlaGymEnv(gym.Env):
    """
    Minimal Carla Gym Environment for PPO training.
    Observation: RGB image (84x84x3) uint8
    Action: 2-dim continuous: [steer (-1..1), throttle (0..1)]
    Episode ends on collision or when max_steps reached.
    """
    def __init__(self,
                 host='127.0.0.1',
                 port=2000,
                 map_name='t3',
                 img_width=84,
                 img_height=84,
                 max_episode_steps=1000,
                 sync=True,
                 spawn_point=None,
                 seed=0):
        super().__init__()
        self.host = host
        self.port = port
        self.map_name = map_name
        self.img_w = img_width
        self.img_h = img_height
        self.max_episode_steps = max_episode_steps
        self.sync = sync
        self.spawn_point = spawn_point
        self.seed = seed

        # Connect to CARLA
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Optionally load a map (commented by default)
        # self.world = self.client.load_world(self.map_name)

        # Set synchronous mode if requested
        if self.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

        # Action space: steer [-1,1], throttle [-1,1] -> we'll map second to [0,1]
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)

        # Observation: image: HxWxC uint8
        self.observation_space = spaces.Box(0, 255, (self.img_h, self.img_w, 3), dtype=np.uint8)

        # Internal state
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.latest_image = None
        self.collision = False
        self.collision_info = None
        self._step_count = 0

        # blueprint library
        self.bp_library = self.world.get_blueprint_library()

        # spawn/respawn
        self._spawn_points = self.world.get_map().get_spawn_points()
        random.seed(self.seed)

    # -------------------------
    # Sensor / spawn utilities
    # -------------------------
    def _destroy_actors(self):
        actors = [self.camera, self.collision_sensor, self.vehicle]
        for a in actors:
            try:
                if a is not None:
                    a.destroy()
            except Exception:
                pass
        self.camera = None
        self.collision_sensor = None
        self.vehicle = None

    def _camera_callback(self, image):
        # Convert CARLA raw image to numpy array (RGB)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        # Resize to desired
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        self.latest_image = rgb

    def _collision_callback(self, event):
        # event: carla.CollisionEvent
        self.collision = True
        other = event.other_actor
        self.collision_info = {
            'frame': event.frame,
            'timestamp': event.timestamp,
            'other_id': other.id,
            'other_type': other.type_id
        }
        # Immediately brake the vehicle so it cannot continue moving
        try:
            if self.vehicle is not None:
                control = carla.VehicleControl()
                control.brake = 1.0
                control.throttle = 0.0
                control.hand_brake = True
                self.vehicle.apply_control(control)
        except Exception:
            pass

    def _spawn_vehicle_and_sensors(self):
        # Destroy previous actors if any
        self._destroy_actors()

        # choose blueprint
        vehicle_bp = self.bp_library.filter('vehicle.*')[0]
        vehicle_bp.set_attribute('role_name', 'rl_agent')

        # choose spawn point
        if self.spawn_point is None:
            spawn_point = random.choice(self._spawn_points)
        else:
            spawn_point = self.spawn_point

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle (maybe spawn point occupied).")

        # Camera sensor
        cam_bp = self.bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.img_w * 2))  # spawn at higher res then resize in callback
        cam_bp.set_attribute('image_size_y', str(self.img_h * 2))
        cam_bp.set_attribute('fov', '90')

        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.camera = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.camera.listen(self._camera_callback)

        # Collision sensor
        col_bp = self.bp_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._collision_callback)

        # small sleep to let sensor initialize (not blocking long)
        # After this, in sync mode, next world.tick() will trigger callbacks
        time.sleep(0.01)

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self):
        # reset flags
        self.collision = False
        self.collision_info = None
        self.latest_image = None
        self._step_count = 0

        # (re)spawn vehicle and sensors
        self._spawn_vehicle_and_sensors()

        # Tick once to populate sensors (in sync mode will call callbacks)
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # wait until an image is available (timeout to avoid deadlock)
        timeout = 2.0
        t0 = time.time()
        while self.latest_image is None and (time.time() - t0) < timeout:
            if self.sync:
                self.world.tick()
            else:
                self.world.wait_for_tick()

        if self.latest_image is None:
            # fallback: return zero image
            obs = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            obs = self.latest_image.copy()

        return obs

    def step(self, action):
        """
        action: np.array([steer in [-1,1], throttle in [-1,1]])
        maps throttle to 0..1 via (a[1] + 1)/2
        """
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        throttle = (throttle + 1.0) / 2.0  # map to 0..1

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = 0.0
        control.hand_brake = False
        self.vehicle.apply_control(control)

        # tick world to advance simulation and trigger sensors
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        self._step_count += 1

        # Build observation
        if self.latest_image is None:
            obs = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            obs = self.latest_image.copy()

        # Reward function (simple): forward speed - small control penalty
        vel = self.vehicle.get_velocity()
        speed = np.linalg.norm([vel.x, vel.y, vel.z])  # m/s
        reward = speed * 0.1  # reward forward motion

        # small steering penalty
        reward -= 0.01 * (abs(steer) + abs(throttle)*0.0)

        done = False
        info = {}

        # collision check
        if self.collision:
            reward -= 100.0
            done = True
            info['collision'] = self.collision_info

        # max steps
        if self._step_count >= self.max_episode_steps:
            done = True
            info['timeout'] = True

        return obs, reward, done, info

    def render(self, mode='human'):
        # rendering handled by CARLA display; nothing here
        pass

    def close(self):
        # clean up actors and restore world settings
        try:
            self._destroy_actors()
            if self.sync:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
        except Exception:
            pass
