import numpy
import tensorflow as tf

from dialog import Dialog


class Encoder(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, batch_size, units, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.units = units
        self.embedding = embedding
        self.gru_layer = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    def call(self, inputs):
        embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
        encoder_output, hidden_state = self.gru_layer(embedded_data,
                                                      initial_state=tf.zeros((self.batch_size, self.units)))
        # <encoder_output> dimension: [batch_size, sequence_len, encoder_units]
        # <encoder_hidden> dimension: [batch_size, encoder_units]
        return encoder_output, hidden_state


class Decoder(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, batch_size, units, embedding, vocab_size, training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.units = units
        self.embedding = embedding
        self.gru_layer = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.vocab_size = vocab_size
        self.training = training

    # <decoder_input>
    # <decoder_hidden> is the last hidden state, carrying information from the last word to the current prediction
    # <encoder_hidden> is the Thought Vector that carries the entire information of the question sentence
    # all these inputs are combined to determine the next prediction of word token
    def call(self, inputs, initial_state):
        if not self.training:      # greedy_algorithm for prediction
            inputs = numpy.asarray([[Dialog.word2id("")]])
            result = numpy.asarray([[]])
            state = initial_state
            for i in range(Dialog.max_dialog_len):
                embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
                decoder_output, decoder_hidden = self.gru_layer(embedded_data,
                                                                initial_state=state)
                decoder_output = self.dense(decoder_output)  # maps the V[unit_size] to V[vocab_size]
                output_word_id = tf.math.argmax(decoder_output, axis=2)
                if output_word_id[0][0].numpy() == 3:
                    val, idx = tf.math.top_k(decoder_output, k=2)
                    output_word_id = [[idx[0][0][1]]]
                # print("Prediction for step: " + str(i))
                # print("Decoder Prediction Input:")
                # print(inputs)
                # print("Decoder Prediction New")
                # print(output_word_id)
                inputs = output_word_id
                result = numpy.append(result, output_word_id)
                state = decoder_hidden
            result = tf.cast(result, tf.dtypes.int32)
            output = [tf.one_hot(result, self.vocab_size, axis=-1)]  # make the output to 3 dimension
            # print("Final Prediction Output")
            # print(output)
        else:
            embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
            decoder_output, decoder_hidden = self.gru_layer(embedded_data,
                                                            initial_state=initial_state)
            output = self.dense(decoder_output)
            epsilon = numpy.full(shape=tf.shape(output), fill_value=0.00001)
            output = output + epsilon
            output = tf.nn.softmax(output, 2)
        return output
