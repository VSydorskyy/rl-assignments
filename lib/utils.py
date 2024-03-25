import gym
from gym.spaces import Box
import numpy as np
import io
import base64
import os
import torch

from gym import wrappers
from matplotlib import pyplot as plt

from .logging import Logger
from .rollout import ReplayBuffer


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super().__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return np.asarray(ob).transpose(*self.op)
    
def make_env(name, seed=None, monitor=False, logdir="./gym-results"):
    env = gym.make(name)
    if monitor:
        env = wrappers.Monitor(env, logdir, force=True)
    if seed:
        env.seed(seed)
    return env

def get_html_video_string(infix, logdir="./gym-results"):
    video = io.open(
        os.path.join(logdir, f'openaigym.video.{infix}.video000000.mp4'),
        'r+b').read()
    encoded = base64.b64encode(video)
    dec_str = encoded.decode('ascii')
    src_tag = f'<source src="data:video/mp4;base64,{dec_str}" type="video/mp4" /></video>'
    html = f'<video width="360" height="auto" alt="test" controls>{src_tag}</video>'
    return html

def train(
    agent,
    env,
    *,
    log_dir,
    prefix,
    buffer_size,
    n_steps,
    warmup_steps,
    target_update_every,
    log_every,
    save_every,
    do_render=False
):
    """Training loop procedure.
    
    Arguments:
        agent: DQN object.
        env: Gym environment.
        log_dir: PathLib path to logging dir.
        prefix: Prefix for the experiment.
        buffer_size: Maximum capacity of the replay buffer.
        n_steps: Total number of training iterations.
        warmup_steps: Number of first iterations that employ random policy.
        target_update_every: Number of iterations between target network updates.
        log_every: Log frequency
        save_every: Save frequency
        
    Returns agent after training.
    """
    
    logger = Logger(log_dir, prefix)
    episode_reward = []
    memory = ReplayBuffer(buffer_size)
    
    obs_cur = env.reset()
    for i in range(n_steps + 1):
        
        if do_render:
            env.render()
        
        if i > warmup_steps:
            obs = torch.FloatTensor(obs_cur).unsqueeze(0)
            act = agent.pick_action(obs)
        else:
            act = env.action_space.sample()

        obs_prev = obs_cur
        obs_cur, rew, done, info = env.step(act)
        episode_reward.append(rew)

        memory.push(obs_prev, act, obs_cur, rew, done)

        if done:
            rew = np.sum(episode_reward)
            episode_reward = []
            logger.log_arr_kv("reward/reward", rew)
            obs_cur = env.reset()

        if i > warmup_steps:

            loss, q, q_est = agent.update_value(memory)
            logger.log_arr_kv("loss/bellman_error_t", loss)
            logger.log_arr_kv("misc/q_t", q)
            logger.log_arr_kv("misc/q_est_t", q_est)

            if i % target_update_every == 0:
                agent.update_target()

            if i % log_every == 0:
                logger.reduce_arr_kv("reward/reward", "reward/reward_mean", np.mean)
                logger.reduce_arr_kv("reward/reward", "reward/reward_max", np.max)
                logger.reduce_arr_kv("reward/reward", "reward/reward_min", np.min)
                logger.reduce_arr_kv("reward/reward", "reward/reward_std", np.std)
                logger.reduce_arr_kv("loss/bellman_error_t", "loss/bellman_error", np.mean)
                logger.reduce_arr_kv("misc/q_t", "misc/q", np.mean)
                logger.reduce_arr_kv("misc/q_est_t", "misc/q_est", np.mean)
                logger.log_kv("misc/epsilon", agent.eps)
                logger.log_kv("misc/timestep", i)
                logger.write_logs(skip_arrs=True)

            if i % save_every == 0:
                num = i // save_every + 1
                save_dir = log_dir / "checkpoints"
                save_dir.mkdir(exist_ok=True)
                agent.save_to(save_dir, prefix=hex(num)[2:].upper())

        agent.update_eps()
    return agent

def plot_progress(data, name=None):
    name = name or "Mean reward"
    plt.plot(data['misc/timestep'], data['reward/reward_mean'], label=name)
    lower = np.array(data['reward/reward_mean'] - data['reward/reward_std']).clip(-1e5, +1e5)
    upper = np.array(data['reward/reward_mean'] + data['reward/reward_std']).clip(-1e5, +1e5)
    examples = np.array(data['misc/timestep'])
    plt.fill_between(list(examples), list(lower), list(upper), color='blue', alpha=0.1)
    plt.legend(loc='best')
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
