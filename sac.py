# -*- encoding: utf-8 -*-
from utils import *
from state_embed import StateEmbed


class SoftAC(object):
    def __init__(self, state_dims, state_embed_dim, action_embed_dim, task_ids, state_embed_hiddens, hiddens,
                 gamma=0.99, actor_lr=0.00025, critic_lr=0.001, tau=0.999, alpha=0.2):
        self.state_dims = state_dims
        self.state_embed_dim = state_embed_dim
        self.action_dim = action_embed_dim
        self.task_ids = task_ids
        self.hiddens = hiddens
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr, self.critic_lr = actor_lr, critic_lr

        self.states_embed = [StateEmbed(dim, state_embed_dim, state_embed_hiddens, task_ids[i]) for i, dim in enumerate(state_dims)]
        self._build()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target)

    def _build(self):
        self.states_pl = [tf.placeholder(tf.float32, shape=(None, dim)) for dim in self.state_dims]
        self.next_states_pl = [tf.placeholder(tf.float32, shape=(None, dim)) for dim in self.state_dims]
        self.a_pl = tf.placeholder(tf.int32, shape=(None,))
        self.a_embed_pl = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.r_pl = tf.placeholder(tf.float32, shape=(None,))
        self.done_pl = tf.placeholder(tf.float32, shape=(None,))
        self.task_id = tf.placeholder(tf.int32)

        # todo: seems tf.case has bugs, cannot use for to generate the cases.
        # todo: Use eager execution to rewrite
        # self.states = tf.case([(tf.equal(self.task_id, i), lambda:self.states_embed[i](self.states_pl[i])) for i in self.task_ids])
        # self.next_state = tf.case([(tf.equal(self.task_id, i), lambda:self.states_embed[i](self.next_states_pl[i])) for i in self.task_ids])

        self.states = tf.case([(tf.equal(self.task_id, 0), lambda: self.states_embed[0](self.states_pl[0])),
                               (tf.equal(self.task_id, 1), lambda: self.states_embed[1](self.states_pl[1])),
                               (tf.equal(self.task_id, 2), lambda: self.states_embed[2](self.states_pl[2])),
                               (tf.equal(self.task_id, 3), lambda: self.states_embed[3](self.states_pl[3]))
                               ])
        next_state = tf.case([(tf.equal(self.task_id, 0), lambda: self.states_embed[0](self.next_states_pl[0])),
                              (tf.equal(self.task_id, 1), lambda: self.states_embed[1](self.next_states_pl[1])),
                              (tf.equal(self.task_id, 2), lambda: self.states_embed[2](self.next_states_pl[2])),
                              (tf.equal(self.task_id, 3), lambda: self.states_embed[3](self.next_states_pl[3]))
                              ])

        # if no state embedding
        # self.states = tf.case([(tf.equal(self.task_id, 0), lambda: self.states_pl[0]),
        #                        (tf.equal(self.task_id, 1), lambda: self.states_pl[1]),
        #                        (tf.equal(self.task_id, 2), lambda: self.states_pl[2])])
        # next_state = tf.case([(tf.equal(self.task_id, 0), lambda: self.next_states_pl[0]),
        #                       (tf.equal(self.task_id, 1), lambda: self.next_states_pl[1]),
        #                       (tf.equal(self.task_id, 2), lambda: self.next_states_pl[2])])

        with tf.variable_scope('main_v'):
            self.v = tf.squeeze(mlp1(self.states, self.hiddens + [1], activation=tf.nn.relu), axis=1)
        with tf.variable_scope('target_v'):
            target_v = tf.squeeze(mlp1(next_state, self.hiddens + [1], activation=tf.nn.relu), axis=1)

        with tf.variable_scope('policy'):
            LOG_STDMIN, LOG_STDMAX = -20, 2
            temp = mlp(self.states, self.hiddens, activation=tf.nn.relu)

            mu = tf.layers.dense(temp, self.action_dim, activation=None)
            log_std = tf.layers.dense(temp, self.action_dim, activation=tf.tanh)
            log_std = LOG_STDMIN + 0.5 * (LOG_STDMAX - LOG_STDMIN) * (log_std + 1)

            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)

            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

            temp = 1 - tf.square(pi)
            clip_up = tf.cast(temp > 1, tf.float32)
            clip_down = tf.cast(temp < 0, tf.float32)
            logp_pi -= tf.reduce_sum(tf.log(temp + tf.stop_gradient(clip_up * (1 - temp) + clip_down * (0 - temp))+ 1e-6), axis=1)

            self.mu = mu
            self.action = pi

        # if train alpha
        # with tf.variable_scope('alpha'):
        #     log_alpha = tf.Variable(0., trainable=True, name='log_alpha')
        #     self.alpha = tf.exp(log_alpha)
        #     alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + 0.2))

        with tf.variable_scope('q1'):
            q1 = tf.squeeze(mlp1(tf.concat([self.states, self.a_embed_pl], axis=-1), self.hiddens + [1],
                                 activation=tf.nn.relu), axis=1)
        with tf.variable_scope('q1', reuse=True):
            q1_pi = tf.squeeze(mlp1(tf.concat([self.states, pi], axis=-1), self.hiddens + [1],
                                    activation=tf.nn.relu), axis=1)

        with tf.variable_scope('q2'):
            q2 = tf.squeeze(mlp1(tf.concat([self.states, self.a_embed_pl], axis=-1), self.hiddens + [1],
                                 activation=tf.nn.relu), axis=1)
        with tf.variable_scope('q2', reuse=True):
            q2_pi = tf.squeeze(mlp1(tf.concat([self.states, pi], axis=-1), self.hiddens + [1],
                                 activation=tf.nn.relu), axis=1)

        q_ = tf.minimum(q1_pi, q2_pi)
        v_ = tf.stop_gradient(q_ - self.alpha * logp_pi)
        target_q = tf.stop_gradient(self.r_pl + self.gamma * (1 - self.done_pl) * target_v)

        v_loss = 0.5 * tf.reduce_mean(tf.square(v_ - self.v))
        self.actor_loss = tf.reduce_mean(self.alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean(tf.square(q1 - target_q))
        q2_loss = 0.5 * tf.reduce_mean(tf.square(q2 - target_q))
        self.critic_loss = v_loss + q1_loss + q2_loss

        actor_opt = tf.train.AdamOptimizer(self.actor_lr)
        critic_opt = tf.train.AdamOptimizer(self.critic_lr)
        # alpha_opt = tf.train.AdamOptimizer(self.critic_lr)

        # self.alpha_train_op = alpha_opt.minimize(alpha_loss)
        self.actor_train_op = actor_opt.minimize(self.actor_loss)  # , var_list=get_vars('policy') + get_vars('state_embed')

        self.critic_train_op = critic_opt.minimize(self.critic_loss)  # , var_list=get_vars('main_v') + get_vars('q') + get_vars('state_embed')

        self.soft_update = [tf.assign(v_targ, self.tau * v_targ + (1 - self.tau) * v)
                            for v, v_targ in zip(get_vars('main_v'), get_vars('target_v'))]

        self.init_target = [tf.assign(v_targ, v) for v, v_targ in zip(get_vars('main_v'), get_vars('target_v'))]

        summaries = []
        summaries.append(tf.summary.scalar('critic_loss', self.critic_loss))
        summaries.append(tf.summary.scalar('actor_loss', self.actor_loss))
        summaries.append(tf.summary.histogram('actions', self.action))
        summaries.append(tf.summary.histogram('state_embedding', self.states))
        self.summary_op = tf.summary.merge(summaries)

    def train(self, step, state, action, action_embed, next_state, reward, done, task_id):
        feeds = {self.states_pl[task_id]: state, self.a_pl: action, self.a_embed_pl: action_embed,
                 self.next_states_pl[task_id]: next_state, self.r_pl: reward, self.done_pl: done, self.task_id: task_id}

        # add data for not used placeholder in states, or it will report error
        for i, dim in enumerate(self.state_dims):
            if i != task_id:
                feeds[self.states_pl[i]] = np.zeros((0, dim))
                feeds[self.next_states_pl[i]] = np.zeros((0, dim))

        # self.sess.run(self.alpha_train_op, feed_dict=feeds)
        actor_loss, _ = self.sess.run([self.actor_loss, self.actor_train_op], feed_dict=feeds)
        summary, critic_loss, _ = self.sess.run([self.summary_op, self.critic_loss, self.critic_train_op], feed_dict=feeds)
        self.sess.run(self.soft_update)
        return actor_loss, critic_loss, summary

    def get_state_embedding(self, state, task_id):
        # process state
        feeds = {self.states_pl[task_id]: state, self.task_id: task_id}
        for i, dim in enumerate(self.state_dims):
            if i != task_id:
                feeds[self.states_pl[i]] = np.zeros((0, dim))
        embed = self.sess.run(self.states, feed_dict=feeds)
        return embed

    def act(self, state, task_id):
        feeds = {self.states_pl[task_id]: state, self.task_id: task_id}
        for i, dim in enumerate(self.state_dims):
            if i != task_id:
                feeds[self.states_pl[i]] = np.zeros((0, dim))
        return self.sess.run(self.action, feed_dict=feeds)

    def save(self, epoch, path="saved_models/sac"):
        self.saver.save(self.sess, path, global_step=epoch)

    def restore(self, path):
        self.saver.restore(self.sess, path)
