import csv
import tensorflow as tf
import numpy

from dialog import Dialog
from model import Encoder, Decoder

# 1. Load Data
print("<<<< Loading Data >>>>")
dialog_list = Dialog.resolve_data("./data/movie_conversations.txt", "./data/movie_lines.txt", vocab_size=2500)
Dialog.load_word2ids()
dialog_x, dialog_y = Dialog.pair_qa_ids(dialog_list)
data_X = tf.data.Dataset.from_tensor_slices(dialog_x)
data_Y = tf.data.Dataset.from_tensor_slices(dialog_y)
data_Y_shifted = data_Y.map(lambda sentence: tf.concat([tf.constant([Dialog.word2id("")]), sentence[:-1]], 0))
print("Amount of observations: " + str(len(dialog_x)))

# 2. Specify Deep Learning Parameters
print("<<<< Specifying Hyper Parameters >>>>")
# the data should not change the parameters significantly, so we are not saving it locally
embedding_dim = 32
vocab_size = len(Dialog.all_tokens) + 4
sequence_len = Dialog.max_dialog_len
batch_size = 32
hidden_units = 64
epoch = 3000
save_interval = 10
learning_rate = 0.005

data_set = tf.data.Dataset.zip((data_X, data_Y, data_Y_shifted))
data_set = data_set.shuffle(buffer_size=len(dialog_x)).batch(batch_size, drop_remainder=True)

print("Vocabulary Size is :" + str(vocab_size))

# 3. Building Model
# using random embedding. after testing, we will try pre-trained word2vec
print("<<<< Building Model >>>>")
def write_table(table, file):
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        [writer.writerow(r) for r in table]


def read_table(file):
    try:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            table = [[float(e) for e in r] for r in reader]
            table = tf.cast(numpy.asarray(table), dtype=tf.dtypes.float32)
            print("Read Saved Embedding")
            return table
    except Exception:
        return None

embedding = read_table("random_embedding.csv")
if embedding is None or len(embedding) < vocab_size:
    print("Rebuild Embedding")
    embedding = tf.random.normal(shape=[vocab_size, embedding_dim])
    write_table(embedding.numpy(), "random_embedding.csv")
encoder = Encoder(batch_size=batch_size, units=hidden_units, embedding=embedding)
# <encoder_output> dimension: [batch_size, sequence_len, encoder_units]; These are useless
# <encoder_hidden> dimension: [batch_size, encoder_units]; This is the THOUGHT VECTOR
decoder = Decoder(batch_size=batch_size, units=hidden_units, embedding=embedding, vocab_size=vocab_size)
# <decoder_output> dimension: [batch_size, sequence_len, encoder_units]
# <decoder_hidden> dimension: [batch_size, encoder_units]
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cross_entropy = tf.keras.losses.CategoricalCrossentropy()

try:
    encoder.load_weights('encoder_weights_saving')
    decoder.load_weights('decoder_weights_saving')
except Exception:
    print("No previous savings found. Start new training.....")


def train(encoder_input, decoder_input, decoder_target):
    loss = 0.0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inputs=encoder_input)
        decoder_predict = decoder(inputs=decoder_input, initial_state=enc_hidden)
        decoder_target = tf.one_hot(decoder_target, vocab_size, axis=-1)
        decoder_predict = tf.cast(decoder_predict, tf.dtypes.float32)
        decoder_target = tf.cast(decoder_target, tf.dtypes.float32)
        loss += cross_entropy(y_true=decoder_target, y_pred=decoder_predict)
        # print("... batch cross entropy loss: " + str(loss.numpy()))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return loss.numpy()


def predict(input_string):
    encoder.batch_size = 1
    decoder.batch_size = 1
    decoder.training = False
    input_batch = []
    sent_ids = Dialog.sent2id(input_string)
    input_batch.append(sent_ids)
    input_np = numpy.asarray(input_batch)
    enc_output, enc_hidden = encoder(inputs=input_np)
    dec_output = decoder(inputs=None, initial_state=enc_hidden)
    output_word_ids = tf.math.argmax(dec_output, axis=2)
    encoder.batch_size = batch_size
    decoder.batch_size = batch_size
    decoder.training = True
    # print(output_word_ids.numpy()[0])
    reply = Dialog.id2sent(output_word_ids.numpy()[0])
    reply = reply.replace("TSTEOSTST", " ]")
    reply = reply.replace("TSTSTARTTST", "[ ")
    return reply


# 4. Run Model
print("<<<< Start Training >>>>")
for step in range(epoch):
    total_loss = 0.0
    num_batches = 0
    for index, (X, Y, Y_shifted) in enumerate(data_set):
        batch_loss = train(X, Y_shifted, Y)
        total_loss += batch_loss
        num_batches += 1
    total_loss = total_loss
    data_set = tf.data.Dataset.zip((data_X, data_Y, data_Y_shifted))
    data_set = data_set.shuffle(buffer_size=len(dialog_x)).batch(batch_size, drop_remainder=True)
    print("Step: " + str(step) + ", Total Loss: " + str(total_loss))
    print("--- Me: [ What's your name? ]")
    print("--- Computer: " + predict("How are you doing, my dear?"))
    if step % save_interval == 0:
        encoder.save_weights('encoder_weights_saving')
        decoder.save_weights('decoder_weights_saving')
        print("Model weights saved ....")
