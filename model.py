import numpy
import tensorflow as tf

from dialog import Dialog


class BeamTreeNode:
    probability = 0.0
    word_id = 0
    first = None;
    second = None;
    third = None;

    def __init__(self, probability, word_id):
        self.probability = probability
        self.word_id = word_id

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
        self.mode = "greedy"

    # <decoder_input>
    # <decoder_hidden> is the last hidden state, carrying information from the last word to the current prediction
    # <encoder_hidden> is the Thought Vector that carries the entire information of the question sentence
    # all these inputs are combined to determine the next prediction of word token
    def call(self, inputs, initial_state):
        if not self.training and self.mode == "greedy":  # greedy algorithm for prediction
            inputs = numpy.asarray([[Dialog.word2id("")]])
            result = numpy.asarray([[]])
            state = initial_state
            for i in range(Dialog.max_dialog_len) and inputs != numpy.asarray([[0], [0], [0]]):
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
        elif not self.training and self.mode == "beam":  # beam search algorithm for prediction, beam size is 3
            inputs = numpy.asarray([[Dialog.word2id("")], [Dialog.word2id("")], [Dialog.word2id("")]])
            state = initial_state
            initial_embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
            initial_decoder_output, initial_decoder_hidden = self.gru_layer(initial_embedded_data,
                                                                            initial_state=state)  # expect this output to be START
            initial_decoder_output = self.dense(initial_decoder_output)
            initial_output_word_id = tf.math.argmax(initial_decoder_output, axis=2)
            result = initial_output_word_id
            inputs = initial_output_word_id
            state = initial_decoder_hidden
            initial_embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
            initial_decoder_output, initial_decoder_hidden = self.gru_layer(initial_embedded_data,
                                                                            initial_state=state)  # getting the first distinct group of outputs
            initial_decoder_output = self.dense(initial_decoder_output)
            epsilon = numpy.full(shape=tf.shape(initial_decoder_output), fill_value=0.00001)
            initial_output = initial_decoder_output + epsilon
            initial_output = tf.nn.softmax(initial_output, 2)
            initial_val, initial_idx = tf.math.top_k(initial_output[0][0], k=3)
            inputs = numpy.asarray([[initial_idx[0]], [initial_idx[1]], [initial_idx[2]]])
            result = numpy.concatenate((result.numpy(), inputs), axis=1)
            prob = tf.math.log(numpy.asarray([initial_val[0], initial_val[1], initial_val[2]]))
            state = initial_decoder_hidden
            for _ in range(Dialog.max_dialog_len) and inputs != numpy.asarray([[0], [0], [0]]):
                embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
                decoder_output, decoder_hidden = self.gru_layer(embedded_data,
                                                                initial_state=state)
                decoder_output = self.dense(decoder_output)  # maps the V[unit_size] to V[vocab_size]
                output = decoder_output + epsilon
                val, idx = tf.math.top_k(output, k=3)
                output_word_id = numpy.asarray([[idx[0][0][0], idx[0][0][1], idx[0][0][2]],
                                                [idx[1][0][0], idx[1][0][1], idx[1][0][2]],
                                                [idx[2][0][0], idx[2][0][1], idx[2][0][2]]])
                output_word_prob = numpy.asarray([[val[0][0][0], val[0][0][1], val[0][0][2]],
                                                  [val[1][0][0], val[1][0][1], val[1][0][2]],
                                                  [val[2][0][0], val[2][0][1], val[2][0][2]]])
                output_word_prob = tf.math.log(output_word_prob)
                prob_sum = numpy.asarray([
                    prob[0] + output_word_prob[0],
                    prob[1] + output_word_prob[1],
                    prob[2] + output_word_prob[2]])
                prob_sum = tf.reshape(prob_sum, shape=[-1])
                val, idx = tf.math.top_k(prob_sum, k=3)
                prob_x = tf.cast(idx / 3, dtype=tf.dtypes.int32)
                prob_y = idx % 3
                adjusted_result = numpy.asarray([
                    result[prob_x[0]],
                    result[prob_x[1]],
                    result[prob_x[2]]
                ])
                adjusted_hidden = numpy.asarray([
                    decoder_hidden[prob_x[0]],
                    decoder_hidden[prob_x[1]],
                    decoder_hidden[prob_x[2]]
                ])
                adjusted_hidden = tf.convert_to_tensor(adjusted_hidden)
                new_tokens = numpy.asarray([
                    [output_word_id[prob_x[0]][prob_y[0]]],
                    [output_word_id[prob_x[1]][prob_y[1]]],
                    [output_word_id[prob_x[2]][prob_y[2]]]
                ])
                inputs = new_tokens
                result = numpy.append(adjusted_result, new_tokens, axis=1)
                state = adjusted_hidden
            result = tf.cast(result[0], tf.dtypes.int32)
            output = [tf.one_hot(result, self.vocab_size, axis=-1)]    # make the output to 3 dimension
        else:
            embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
            decoder_output, decoder_hidden = self.gru_layer(embedded_data,
                                                            initial_state=initial_state)
            output = self.dense(decoder_output)
            epsilon = numpy.full(shape=tf.shape(output), fill_value=0.00001)
            output = output + epsilon
            output = tf.nn.softmax(output, 2)
        return output
