import tensorflow as tf


class StateEmbed(object):
    def __init__(self, state_dim, embed_dim, hiddens, task_id):
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hiddens = hiddens
        self.task_id = task_id

    def forward(self, x):
        with tf.variable_scope('state_embed_{}'.format(self.task_id), reuse=tf.AUTO_REUSE):
            for h in self.hiddens:
                x = tf.layers.dense(x, h, activation=tf.nn.relu)
            x = tf.layers.dense(x, self.embed_dim)
            # x = tf.contrib.layers.layer_norm(x)
        return x

    def __call__(self, x):
        return self.forward(x)

0