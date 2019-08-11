import csv
import numpy
import tensorflow as tf

from dialog import Dialog
from model import Encoder, Decoder

print("Loading Resources")


def predict(input_string):
    sent_ids = Dialog.sent2id(input_string)
    input_batch = [sent_ids]
    input_np = numpy.asarray(input_batch)
    enc_output, enc_hidden = encoder(inputs=input_np)
    enc_hidden = numpy.asarray([enc_hidden[0].numpy(), enc_hidden[0].numpy(), enc_hidden[0].numpy()])
    enc_hidden = tf.convert_to_tensor(enc_hidden)
    dec_output = decoder(inputs=None, initial_state=enc_hidden)
    output_word_ids = tf.math.argmax(dec_output, axis=2)
    # print(output_word_ids.numpy()[0])
    reply = Dialog.id2sent(output_word_ids.numpy()[0])
    reply = reply.replace("TSTEOSTST", "")
    reply = reply.replace("TSTSTARTTST", "")
    return reply


def read_table(file):
    try:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            table = [[float(e) for e in r] for r in reader]
            table = tf.cast(numpy.asarray(table), dtype=tf.dtypes.float32)
            return table
    except Exception:
        return None


embedding = read_table("random_embedding.csv")
dialog_list = Dialog.resolve_data("./data/movie_conversations.txt", "./data/movie_lines.txt", vocab_size=800)
Dialog.load_word2ids()
encoder = Encoder(batch_size=1, units=64, embedding=embedding)
decoder = Decoder(batch_size=3, units=64, embedding=embedding, vocab_size=804)
encoder.load_weights('encoder_weights_saving')
decoder.load_weights('decoder_weights_saving')
decoder.training = False
decoder.mode = "beam"

print("<<< Chatting Bot Demonstration >>>")
print("Mode: 3 - Beam Search")
val = ""
while val != "Exit":
    val = input("Me: ")
    inp = str("TSTSTARTTST " + val + " TSTEOSTST")
    rep = predict(inp)
    print("Computer: " + rep)
