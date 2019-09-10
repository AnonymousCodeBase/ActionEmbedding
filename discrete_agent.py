import numpy as np
import tensorflow as tf
from sac_discrete import DiscreteSoftAC
from config import Config
from experience import ExperienceCell
import argparse
import gym
import os
from envs.mujoco_wrapper import MujocoWrapper
from envs.grid_world import GridWorld


class DiscreteAgent(object):
    def __init__(self, env, state_dim, act_dim):
        self.env = env
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.policy_net = DiscreteSoftAC(state_dim, act_dim, Config.ac_hiddens, Config.state_embed_hiddens, Config.gamma, Config.actor_lr,
                                         Config.critic_lr, Config.tau, Config.alpha)

        self.experience = ExperienceCell(state_dim, 1, Config.seq_len, Config.memory_size)
        self.writer = None
        if args.summary:
            self.writer = tf.summary.FileWriter(Config.summary_folder)

    def choose_action(self, state, determinitic=False, random=False):
        if random:
            a = np.random.choice(self.env.actions)
        else:
            a = self.policy_net.act(state.reshape(1, -1).astype(np.float32))[0]
        return a

    def train_policy(self, epoch):
        s, a, a_embed, n_s, r, d = self.experience.sample(Config.batch_size)
        loss_act, loss_crt, summary = self.policy_net.train(epoch, s, a, n_s, r, d)
        if self.writer:
            self.writer.add_summary(summary, global_step=epoch)
        return loss_act, loss_crt

    def train(self):
        global_step = 0
        rewards = []
        for i in range(Config.epoches):
            env = self.env
            env.reset()
            total_r, done, step = 0, False, 0
            while not done and step < Config.max_step:  #  // (task_id + 1)
                obs = env.get_state()
                a = self.choose_action(obs)
                n_obs, r, done, _ = env.step(a)
                total_r += r
                step += 1
                self.experience.store(obs, a, np.zeros(1), n_obs, r, done)
                # self.envs[task_id].render()

                if self.experience.cur_size > Config.batch_size * 2:
                    self.train_policy(global_step)
                if step % (Config.seq_len + 1) == 0:
                    self.experience.finish()
                global_step += 1
            self.experience.finish()
            print('epoch: {}, steps: {} total reward: {}'.format(i, step, total_r))
            rewards.append(total_r)
            if len(rewards) > 100:
                rewards.pop(0)
            avg_reward = np.mean(rewards)

            if self.writer:
                summary = tf.Summary(value=[tf.Summary.Value(tag="reward1", simple_value=avg_reward)])
                self.writer.add_summary(summary, global_step=global_step)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='0', type=str)
    parser.add_argument('-t', default='0', type=int)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-summary', default=False, type=bool)
    return parser.parse_args()


def save_rewards(path, rewards):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + "rewards_" + str(args.seed), rewards)


if __name__ == '__main__':
    args = parse()

    # np.random.seed(args.seed)
    # tf.random.set_random_seed(args.seed)

    # Config.summary_folder = Config.summary_folder + "_" + args.i + "/" + str(args.seed)

    # seq = int(args.t) + 1
    # env = GridWorld(seq)
    # agent = DiscreteAgent(env, 4, 4 ** seq)

    envs = [MujocoWrapper('InvertedPendulum-v2', 101), MujocoWrapper('RoboschoolInvertedPendulum-v1', 91),
            MujocoWrapper('InvertedDoublePendulum-v2', 51), MujocoWrapper('RoboschoolInvertedDoublePendulum-v1', 71)]
    action_nums = [101, 91, 51, 71]

    env = envs[args.t]
    agent = DiscreteAgent(env, env.observation_space.shape[0], action_nums[args.t])

    agent.train()
