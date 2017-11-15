import tensorflow as tf
import numpy as np
import math
import sys
from seq2seq import Seq2Seq
from dialog import Dialog


class ChatBot:

    def __init__(self, voc_path):
        self.dialog = Dialog()
        self.dialog.load_vocab(voc_path)

        self.model = Seq2Seq(self.dialog.vocab_size)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.max_decode_len = 20

    def run(self):
        self.saver.restore(self.sess, 'model')
        sys.stdout.write("> ")
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            print(self.get_replay(line.strip()))

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()

    def decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist()

        # TODO: 구글처럼 시퀀스 사이즈에 따라 적당한 버킷을 사용하도록 만들어서 사용하도록
        input_len = int(math.ceil((len(enc_input) + 1) * 1.5))

        enc_input, dec_input, _ = self.dialog.transform(enc_input, dec_input,
                                                        input_len,
                                                        self.max_decode_len)

        return self.model.predict(self.sess, [enc_input], [dec_input])

    def get_replay(self, msg):
        enc_input = self.dialog.tokenizer(msg)
        enc_input = self.dialog.tokens_to_ids(enc_input)
        dec_input = []

        curr_seq = 0
        for i in range(self.max_decode_len):
            outputs = self.decode(enc_input, dec_input)
            if self.dialog.is_eos(outputs[0][curr_seq]):
                break
            elif self.dialog.is_defined(outputs[0][curr_seq]) is not True:
                dec_input.append(outputs[0][curr_seq])
                curr_seq += 1

        reply = self.dialog.decode([dec_input], True)

        return reply


def main(_):
    print("start chat!!\n")

    chatbot = ChatBot("./data/word")
    chatbot.run()


if __name__ == "__main__":
    tf.app.run()
