# TODO :  S2S 모델은 학습 말뭉치의 단어 수가 분류해야할 클래스의 개수가 되기 때문에 단어수가 지나치게 크면 학습의 질은 물론 속도도 급격하게 감소하는 문제가 있습니다.
# TODO :  단어수가 지나치게 크면 학습의 질은 물론 속도도 급격하게 감소하는 문제가 있다.

# TODO : decoder 에 넣을 최대 sequece 수는 단어 20개
import tensorflow as tf


# Seq2Seq 기본 클래스
class Seq2Seq:

    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # TODO : 엔코더
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        # TODO : 디코더
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size])
        self.targets = tf.placeholder(tf.int64, [None, None])

        self.weights = tf.Variable(tf.ones([self.n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.build_model()

        self.saver = tf.train.Saver()

    def build_model(self):
        self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        self.dec_input = tf.transpose(self.dec_input, [1, 0, 2])

        # TODO : go to build cells
        enc_cell, dec_cell = self.build_cells()

        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32)

        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32,
                                                    initial_state=enc_states)

        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.targets)

        self.outputs = tf.argmax(self.logits, 2)

    def cell(self, n_hidden, output_keep_prob):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden) # n_hidden = 128
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob) # rnn 에 dropout을 적용
        return rnn_cell

    # TODO : cell 은 멀티 레이어 층을 가지고 있는 RNN
    def build_cells(self, output_keep_prob=0.5):
        # TODO : encoder cell
        enc_cell = tf.contrib.rnn.MultiRNNCell([self.cell(self.n_hidden, output_keep_prob)
                                                for _ in range(self.n_layers)])
        # TODO : decoder cell
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.cell(self.n_hidden, output_keep_prob)
                                                for _ in range(self.n_layers)])

        return enc_cell, dec_cell

    def build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        # TODO : loss 를 seq2seq2 로스를 사용하지 않았다.
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    # TODO : test
    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})
