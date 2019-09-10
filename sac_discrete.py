# -*- encoding: utf-8 -*-
from utils import *
from state_embed import StateEmbed


class DiscreteSoftAC(object):
    def __init__(self, state_dim, action_dim, hiddens, state_embed_hiddens,
                 gamma=0.99, actor_lr=0.00025, critic_lr=0.001, tau=0.999, alpha=0.2):
        self.state_dim = state_dim
        self.state_embed_dim = 5
        self.action_dim = action_dim
        self.hiddens = hiddens
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha = 0
        self.actor_lr, self.critic_lr = actor_lr, critic_lr

        # self.state_embed = StateEmbed(state_dim, 5, state_embed_hiddens, 0)

        self._build()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target)

    def _build(self):
        self.states_pl = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.next_states_pl = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.a_pl = tf.placeholder(tf.int32, shape=(None,))
        self.r_pl = tf.placeholder(tf.float32, shape=(None,))
        self.done_pl = tf.placeholder(tf.float32, shape=(None,))

        self.state = self.states_pl
        self.next_state = self.next_states_pl

        # self.state = self.state_embed(self.states_pl)
        # self.next_state = self.state_embed(self.next_states_pl)

        with tf.variable_scope('main_v'):
            self.v = tf.squeeze(mlp1(self.state, self.hiddens + [1], activation=tf.nn.relu), axis=1)
        with tf.variable_scope('target_v'):
            target_v = tf.squeeze(mlp1(self.next_state, self.hiddens + [1], activation=tf.nn.relu), axis=1)

        with tf.variable_scope('policy'):
            temp = mlp(self.state, self.hiddens, activation=tf.nn.relu)
            action_logit = tf.layers.dense(temp, self.action_dim, activation=None)
            probs = tf.nn.softmax(logits=action_logit, axis=1)
            logp_probs = tf.nn.log_softmax(logits=action_logit, axis=1)
            self.max_prob_action = tf.argmax(action_logit, axis=1)
            pi = tf.squeeze(tf.multinomial(action_logit, 1), axis=1)

            # target_distribution = tf.distributions.Categorical(logits=self.alpha * q_min)
            one_hot_pi = tf.one_hot(pi, depth=self.action_dim)
            logp_pi = tf.reduce_sum(logp_probs * one_hot_pi, axis=1)

            self.action = pi

        # with tf.variable_scope('alpha'):
        #     log_alpha = tf.Variable(0., trainable=True, name='log_alpha')
        #     self.alpha = tf.exp(log_alpha)
        #     alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + 0.2))

        with tf.variable_scope('q1'):
            q1_all = mlp1(self.state, self.hiddens + [self.action_dim], activation=tf.nn.relu)

        with tf.variable_scope('q2'):
            q2_all = mlp1(self.state, self.hiddens + [self.action_dim], activation=tf.nn.relu)

        actions = tf.one_hot(self.a_pl, depth=self.action_dim)
        q1 = tf.reduce_sum(q1_all * actions, axis=1)
        q2 = tf.reduce_sum(q2_all * actions, axis=1)
        q1_pi = tf.reduce_sum(q1_all * one_hot_pi, axis=1)
        q2_pi = tf.reduce_sum(q2_all * one_hot_pi, axis=1)

        q_ = tf.minimum(q1_pi, q2_pi)
        v_ = tf.stop_gradient(q_ - self.alpha * logp_pi)
        target_q = tf.stop_gradient(self.r_pl + self.gamma * (1 - self.done_pl) * target_v)


        # refer to the code below
        # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
        # min_q = tf.minimum(q1_all, q2_all)
        inside = self.alpha * logp_probs - q1_all
        self.actor_loss = tf.reduce_mean(probs * inside)

        v_loss = 0.5 * tf.reduce_mean(tf.square(v_ - self.v))
        q1_loss = 0.5 * tf.reduce_mean(tf.square(q1 - target_q))
        q2_loss = 0.5 * tf.reduce_mean(tf.square(q2 - target_q))
        self.critic_loss = v_loss + q1_loss + q2_loss

        # alpha_opt = tf.train.AdamOptimizer(self.critic_lr)
        actor_opt = tf.train.AdamOptimizer(self.actor_lr)
        critic_opt = tf.train.AdamOptimizer(self.critic_lr)

        # self.alpha_train_op = alpha_opt.minimize(alpha_loss)
        self.actor_train_op = actor_opt.minimize(self.actor_loss, var_list=get_vars('policy'))  # , var_list=get_vars('policy') + get_vars('state_embed')

        self.critic_train_op = critic_opt.minimize(self.critic_loss, var_list=get_vars('main_v') + get_vars('q'))  # , var_list=get_vars('main_v') + get_vars('q') + get_vars('state_embed')

        self.soft_update = [tf.assign(v_targ, self.tau * v_targ + (1 - self.tau) * v)
                            for v, v_targ in zip(get_vars('main_v'), get_vars('target_v'))]

        self.init_target = [tf.assign(v_targ, v) for v, v_targ in zip(get_vars('main_v'), get_vars('target_v'))]

        tf.summary.scalar('critic_loss', self.critic_loss)
        tf.summary.scalar('actor_loss', self.actor_loss)
        tf.summary.histogram('actions', self.action)
        self.summary_op = tf.summary.merge_all()

    def train(self, step, state, action, next_state, reward, done):
        feeds = {self.states_pl: state, self.a_pl: action,
                 self.next_states_pl: next_state, self.r_pl: reward, self.done_pl: done}

        # self.sess.run(self.alpha_train_op, feed_dict=feeds)
        actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train_op], feed_dict=feeds)
        summary, critic_loss, _ = self.sess.run([self.summary_op, self.critic_loss, self.critic_train_op], feed_dict=feeds)
        self.sess.run(self.soft_update)

        return actor_loss, critic_loss, summary

    def act(self, state):
        feeds = {self.states_pl: state}
        a = self.sess.run(self.action, feed_dict=feeds)

        return a

    def save(self, epoch, path="saved_models/sac"):
        self.saver.save(self.sess, path, global_step=epoch)

    def restore(self, path):
        self.saver.restore(self.sess, path)
