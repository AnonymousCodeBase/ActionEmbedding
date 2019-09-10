import numpy as np
import tensorflow as tf
import gym
import os
import argparse
from envs.mujoco_wrapper import MujocoWrapper
from envs.grid_world import GridWorld
from sklearn.metrics.pairwise import pairwise_distances
from action_embed import ActionEmbed
from sac import SoftAC
from config import Config
from experience import Expericence


class Agent(object):
    def __init__(self, envs):
        self.envs = envs
        self.task_ids = np.arange(len(envs), dtype=int)

        self.noise_std = 1
        self.noise_decay = 0.99999
        self.noise_min = 0.001

        self.policy_net = SoftAC(Config.state_dims, Config.state_embed_dim, Config.action_embed_dim, self.task_ids, Config.state_embed_hiddens,
                                      Config.ac_hiddens, Config.gamma, Config.actor_lr, Config.critic_lr, Config.tau, Config.alpha)

        self.action_embed = ActionEmbed(Config.state_embed_dim, Config.action_embed_dim, Config.action_dims, self.task_ids,
                                        Config.seq_len, Config.action_embed_hiddens, Config.cell_num, Config.action_embed_lr)

        self.experiences = Expericence(Config.state_dims, Config.action_embed_dim, Config.seq_len, self.task_ids, Config.memory_size)
        # initialize the embedding
        self.action_embeddings = self.action_embed.get_embedding()
        self.writer = None
        if args.summary:
            self.writer = tf.summary.FileWriter(Config.summary_folder)

    def choose_action(self, state, task_id, random=False):
        if random:
            a = np.random.choice(self.envs[task_id].actions)
            return a, self.action_embeddings[task_id][a]
        a_hat = self.policy_net.act(state.reshape(1, -1).astype(np.float32), task_id)

        embed_mat = self.action_embeddings[task_id]
        a = self.nearest(a_hat, embed_mat)
        return a, a_hat

    def nearest(self, a, embedding):
        distance = pairwise_distances(a, embedding)[0]
        closest = np.argmin(distance)
        return closest

    def train_policy(self, epoch, task_id):
        s, a, a_embed, n_s, r, d = self.experiences.sample(Config.batch_size, task_id)
        loss_act, loss_crt, summary = self.policy_net.train(epoch, s, a, a_embed, n_s, r, d, task_id)  # self.action_embeddings[task_id][a]
        if self.writer:
            self.writer.add_summary(summary, global_step=epoch)

        # print('trainning policy. epoch {}: actor loss: {}, critic loss: {}'.format(epoch, loss_act, loss_crt))
        return loss_act, loss_crt

    def train_embedding(self, epoch, task_id):
        states, actions, length = self.experiences.sample_traj(Config.action_batch_size, task_id)
        shape = states.shape
        state_embed = self.policy_net.get_state_embedding(states.reshape(-1, shape[-1]), task_id)
        state_embed = state_embed.reshape(shape[0], shape[1], -1)

        # if no state embedding for same state space
        # state_embed = states
        loss, summary = self.action_embed.train(epoch, state_embed, actions[:, :-1], length, task_id)

        # update current embeddings
        self.action_embeddings = self.action_embed.get_embedding()

        if self.writer:
            self.writer.add_summary(summary, global_step=epoch)

        # print('trainning embedding. epoch {}: embedding loss: {}'.format(epoch, loss))
        return loss

    def train(self, tasks=(1,)):
        global_step = 0
        rewards = []

        if not os.path.exists('data/{}/'.format(args.i)):
            os.makedirs('data/{}/'.format(args.i))

        for i in range(Config.epoches):
            # todo: may need to modify
            task_id = np.random.choice(tasks)
            env = self.envs[task_id]
            env.reset()
            total_r, done, step = 0, False, 0
            while not done and step < Config.max_step:  #  // (task_id + 1)
                obs = env.get_state()
                a, a_hat = self.choose_action(obs, task_id)
                n_obs, r, done, _ = env.step(a)
                # self.envs[task_id].render()
                total_r += r
                step += 1

                self.experiences.store(obs, a, a_hat, n_obs, r, done, task_id)

                # train policy
                if self.experiences.get_size(task_id) > Config.batch_size * 2:
                    self.train_policy(global_step, task_id)

                # train embedding
                if self.experiences.get_traj_size(task_id) > Config.action_batch_size:
                    embed_loss = self.train_embedding(global_step, task_id)

                # if reach the maximum length of trajectory in the replay memory
                if step % Config.seq_len == 0:
                    self.experiences.finish(task_id)

                global_step += 1
                if global_step % 1000 == 0:
                    self.policy_net.save(global_step, path=Config.model_save_folder + task_name + "/" +str(args.seed) + "/sac")
                    self.action_embed.save(global_step, path=Config.model_save_folder + task_name + "/" +str(args.seed) + "/embedding")

            self.experiences.finish(task_id)
            print('epoch: {}, task id: {}, steps: {} total reward: {}'.format(i, task_id, step, total_r))

            rewards.append(total_r)
            if len(rewards) > 100:
                rewards.pop(0)
            avg_reward = np.mean(rewards)

            summary = tf.Summary(value=[tf.Summary.Value(tag="reward1", simple_value=avg_reward)])

            if self.writer:
                self.writer.add_summary(summary, global_step=global_step)

    def transfer(self, task_id, embedding_path=None, policy_path=None):
        if embedding_path:
            self.action_embed.restore(embedding_path)
        if policy_path:
            self.policy_net.restore(policy_path)

        self.action_embeddings = self.action_embed.get_embedding()
        self.train([task_id])


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='0', type=str)
    parser.add_argument('-t', default=0, type=int)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-summary', default=True, type=bool)
    parser.add_argument('-ckpt_path', type=str, required=False)
    parser.add_argument('-ckpt_step', type=str, required=False)
    parser.add_argument('-source_t', type=int, required=False)
    parser.add_argument('-transfer', default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse()
    print(args)
    Config.summary_folder = Config.summary_folder + args.i
    # np.random.seed(args.seed)
    # tf.random.set_random_seed(args.seed)

    all_task_name = ['gridworld-seq1', 'gridworld-seq2', 'gridworld-seq3']
    # all_task_name = ['Pendulum', 'RoboschoolPendulum', 'DoublePendulum', 'RoboschoolDoublePendulum']

    if args.transfer == 1:
        task_name = 'transfer-{}-{}'.format(args.source_t, args.t)
    else:
        task_name = all_task_name[args.t]

    if not os.path.exists(Config.model_save_folder + task_name + "/" +str(args.seed)):
        os.makedirs(Config.model_save_folder + task_name + "/" +str(args.seed))

    # GridWorld Settings
    # envs = [GridWorld(1), GridWorld(2), GridWorld(3)]
    # Config.state_dims = [4, 4, 4]
    # Config.action_dims = [4, 16, 64]
    # agent = Agent(envs)

    # DoublePendulum in Mujoco and RoboSchool Settings
    envs = [MujocoWrapper('InvertedPendulum-v2', 101), MujocoWrapper('RoboschoolInvertedPendulum-v1', 91),
            MujocoWrapper('InvertedDoublePendulum-v2', 51), MujocoWrapper('RoboschoolInvertedDoublePendulum-v1', 71)]
    Config.state_dims = [env.observation_space.shape[0] for env in envs]
    Config.action_dims = [101, 91, 51, 71]
    agent = Agent(envs)

    if args.transfer == 1:
        agent.transfer(args.t, embedding_path=args.ckpt_path + "/embedding-" + args.ckpt_step,
                       policy_path=args.ckpt_path + "/sac-" + args.ckpt_step)
    else:
        agent.train([args.t])
