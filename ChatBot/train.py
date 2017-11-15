# TODO: 전반적인 문제점 loss 가 흔들린다. learning rate 조정해야 함.
# TODO: seq2seq 모델 구조 = 인코더 입력 : Good morning! 디코더 입력 : <go> 좋은 아침입니다! 디코더 출력 : 좋은 아침입니다! <eos>
# 인코더는 압축, 디코더는 압축된 정보를 받아서 타겟으로 변환해 출력.
import tensorflow as tf
import math
from seq2seq import Seq2Seq
from dialog import Dialog

def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Learning start.")
        sess.run(tf.global_variables_initializer())

        # TODO : math.ceil 은 올림함수 이다. ex ) 3.22 -> 4

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))

        # step = 853 * 1000 = 853000
        for step in range(total_batch * epoch):

            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 100 == 0:

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

        saver.save(sess, 'model')
        print('Finish training .')

        enc_input, dec_input, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)

        print("\nAccuracy :", accuracy)


def main(_):

    dialog = Dialog()

    dialog.load_vocab("./data/word")
    dialog.load_examples("./data/chat")


    train(dialog, batch_size=100, epoch= 1000)


if __name__ == "__main__":
    tf.app.run()
