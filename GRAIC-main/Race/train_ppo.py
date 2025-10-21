# train_ppo.py
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import numpy as np
from ppo import CarlaGymEnv
import os

def make_env(args):
    def _init():
        env = CarlaGymEnv(host=args.host,
                          port=args.port,
                          map_name=args.map,
                          img_width=args.img_w,
                          img_height=args.img_h,
                          max_episode_steps=args.max_steps,
                          sync=True,
                          seed=args.seed)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--map', default='Town03')
    parser.add_argument('--img-w', default=84, type=int)
    parser.add_argument('--img-h', default=84, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--total-timesteps', default=200000, type=int)
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--max-steps', default=1000, type=int)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args)])
    # SB3 expects channels-first image: (C,H,W)
    env = VecTransposeImage(env)  # convert HWC to CHW automatically

    # policy_kwargs can be tuned
    model = PPO('CnnPolicy', env,
                verbose=1,
                tensorboard_log=args.log_dir,
                seed=args.seed,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                learning_rate=2.5e-4)

    # checkpoint callback every N steps
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=args.log_dir,
                                             name_prefix='ppo_carla')

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        model.save(os.path.join(args.log_dir, 'ppo_carla_final'))
        print("Saved model to", args.log_dir)

if __name__ == '__main__':
    main()
