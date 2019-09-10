import tensorflow as tf
from utils import *


class ActionEmbed(object):
    def __init__(self, state_embed_dim, act_embed_dim, action_nums, task_ids, seq_len, hiddens, cell_num,
                 lr=0.001):
        self.state_dim = state_embed_dim
        self.act_dim = act_embed_dim
        self.action_nums = action_nums
        self.task_ids = task_ids
        self.seq_len = seq_len
        self.hiddens = hiddens
        self.cell_num = cell_num
        self.lr = lr

        self._build()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build(self):
        self.state_pl = tf.placeholder(tf.float32, shape=(None, self.seq_len + 1, self.state_dim))
        self.a_pl = tf.placeholder(tf.int32, shape=(None, self.seq_len))
        self.length_pl = tf.placeholder(tf.int32, shape=(None,))
        self.task_id = tf.placeholder(tf.int32)
        self.all_pl = [self.state_pl, self.a_pl, self.length_pl, self.task_id]
        with tf.variable_scope('action_embedding'):
            self.embeddings = [tf.get_variable('embedding_{}'.format(self.task_ids[i]), shape=(n, self.act_dim),
                                               initializer=tf.orthogonal_initializer) for i, n in enumerate(self.action_nums)]

        states = self.state_pl[:, :-1, :]
        targets = self.state_pl[:, 1:, :]

        # todo: seems tf.case has bugs, cannot use for to generate the cases.
        # todo: Use eager execution to rewrite
        # embedding_mat = tf.case([(tf.equal(self.task_id, i), lambda:self.embeddings[i]) for i in self.task_ids])
        embedding_mat = tf.case([(tf.equal(self.task_id, 0), lambda: self.embeddings[0]),
                                 (tf.equal(self.task_id, 1), lambda: self.embeddings[1]),
                                 (tf.equal(self.task_id, 2), lambda: self.embeddings[2]),
                                 (tf.equal(self.task_id, 3), lambda: self.embeddings[3])
                                 ])

        act_embed = tf.nn.embedding_lookup(embedding_mat, self.a_pl)

        inputs = tf.concat([states, act_embed], axis=2)

        rnncell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_num)
        # rnn = tf.nn.rnn_cell.DropoutWrapper(rnncell, output_keep_prob=0.9)
        outputs, s = tf.nn.dynamic_rnn(rnncell, inputs, self.length_pl, dtype=tf.float32)

        x = tf.reshape(outputs, (-1, self.cell_num))
        x = mlp(x, self.hiddens, activation=tf.nn.relu)
        x = tf.layers.dense(x, self.state_dim, activation=None)

        targets = tf.reshape(targets, (-1, self.state_dim))
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - targets), axis=1))
        optimizers = tf.train.AdamOptimizer(self.lr)
        self.train_all_op = optimizers.minimize(self.loss)
        self.train_embed_op = optimizers.minimize(self.loss, var_list=get_vars('action_embedding'))

        summaries = [tf.summary.histogram('embedding_{}'.format(i), self.embeddings[i]) for i in range(len(self.action_nums))]
        summaries.append(tf.summary.scalar('embedding_loss', self.loss))
        self.summary_op = tf.summary.merge(summaries)

    def train(self, step, states, actions, length, task_id):
        feeds = {k: v for k, v in zip(self.all_pl, [states, actions, length, task_id])}
        loss, summary, _ = self.sess.run([self.loss, self.summary_op, self.train_all_op], feed_dict=feeds)
        return loss, summary

    def train_embedding(self, step, states, actions, length, task_id):
        feeds = {k: v for k, v in zip(self.all_pl, [states, actions, length, task_id])}
        self.sess.run(self.train_embed_op, feed_dict=feeds)

    def get_embedding(self):
        return self.sess.run(self.embeddings)

    def save(self, epoch, path="saved_models/action_embedding"):
        self.saver.save(self.sess, path, global_step=epoch)

    def restore(self, path):
        self.saver.restore(self.sess, path)
