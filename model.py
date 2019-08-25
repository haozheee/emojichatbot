import numpy
import tensorflow as tf

from dialog import Dialog


class Attention(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = tf.keras.layers.Dense(1)

    # encoder_output is batch_size * sequence_len * hidden_size
    # decoder_hidden is batch_size * hidden_size
    @tf.function
    def call(self, encoder_output, decoder_hidden):
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)  # batch_size * 1 * decoder_hidden_size
        decoder_hidden = tf.tile(decoder_hidden, multiples=[1, Dialog.max_dialog_len, 1])
        concat_hidden = tf.concat([encoder_output, decoder_hidden], axis=2)  # batch_size * sequence_len * 2hidden_size
        score = self.dense(concat_hidden)  # batch_size * sequence_len * 1
        score = tf.squeeze(score) # batch_size * sequence_len
        score = tf.math.softmax(score)
        # print("Score is: ")
        # print(score)
        alignment = tf.einsum("ijk,ij->ijk", encoder_output, score)
        # print("Alignment Vector is: ")  # batch_size * sequence_len * hidden_size
        # print(alignment)
        context_vector = tf.einsum("ijk->ik", alignment)
        context_vector = tf.expand_dims(context_vector, 1)
        # print("Context Vector is :")
        # print(context_vector)
        return context_vector  # batch_size * 1 * hidden_size

class Encoder(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, batch_size, units, embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.units = units
        self.embedding = embedding
        self.gru_layer = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

    @tf.function
    def call(self, inputs):
        embedded_data = tf.nn.embedding_lookup(self.embedding, inputs)
        encoder_output, hidden_state = self.gru_layer(embedded_data,
                                                      initial_state=tf.zeros((self.batch_size, self.units)))
        # encoder hidden state is : batch_size * hidden_unit
        # <encoder_output> dimension: [batch_size, sequence_len, encoder_units]
        # <encoder_hidden> dimension: [batch_size, encoder_units]
        # print("Encoder Output:")
        # print(encoder_output)
        # print("Encoder Hidden:")
        # print(hidden_state)
        return encoder_output, hidden_state


class Decoder(tf.keras.Model):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, batch_size, embedding, vocab_size, encoder_hidden_size, embedding_dim, training=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.encoder_hidden_size = encoder_hidden_size
        self.embedding_dim = embedding_dim
        self.embedding = embedding
        self.gru_layer = tf.keras.layers.GRU(encoder_hidden_size + embedding_dim, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        # maps encoder_hidden_size to decoder_hidden_size: (embedding_dim+hidden_size)
        self.fc = tf.keras.layers.Dense(encoder_hidden_size + embedding_dim)
        self.vocab_size = vocab_size
        self.training = training
        self.attention = Attention()
        self.mode = "greedy"

    # <inputs> is the input matrix of word ids to the decoder, which is batch_size * sequence_len
    # <initial_state> is the initial hidden state of decoder
    # <encoder_output> is the output from the encoder, which is batch_size * sequence_len * hidden_size
    # all these inputs are combined to determine the next prediction of word token
    @tf.function
    def call(self, inputs, initial_state, encoder_output):
        # batch_size * enc_hidden_size -> batch_size * dec_hidden_size: embedding_dim+hidden_size
        initial_state = self.fc(initial_state)
        if not self.training and self.mode == "greedy":  # greedy algorithm for prediction
            next_input = numpy.asarray([[Dialog.word2id("TSTSTARTTST")]])
            result = numpy.asarray([[]])
            state = initial_state
            for i in range(Dialog.max_dialog_len) and next_input != numpy.asarray([[0], [0], [0]]):
                embedded_data = tf.nn.embedding_lookup(self.embedding, next_input)  # batch * 1 * embedding_dim
                context_vec = self.attention(encoder_output, state)
                concat_input = tf.concat(embedded_data, context_vec, 2) #batch * 1 * (embedding_dim+enc_hidden_size)
                decoder_output, decoder_hidden = self.gru_layer(concat_input,
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
                next_input = output_word_id
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
            output = [tf.one_hot(result, self.vocab_size, axis=-1)]  # make the output to 3 dimension
        elif self.training:
            output = None
            state = initial_state
            # select the 0th token from observations amount of batch_size
            next_input = tf.slice(inputs, begin=[0, 0], size=[-1, 1])
            for i in range(Dialog.max_dialog_len):
                embedded_data = tf.nn.embedding_lookup(self.embedding, next_input)  # batch * 1 * embedding_dim
                context_vec = self.attention(encoder_output, state)
                # print(embedded_data)
                # print(context_vec)
                concat_input = tf.concat([embedded_data, context_vec], 2)  # batch * 1 * (embedding_dim+enc_hidden_size)
                # print(tf.shape(concat_input))
                # print(tf.shape(state))
                decoder_output, decoder_hidden = self.gru_layer(inputs=concat_input,   # batch * 1 * (embedding_dim+enc_hidden_size)
                                                                initial_state=state) # batch * (embedding_dim+enc_hidden_size)
                decoder_output = self.dense(decoder_output)  # batch * 1 * vocab_size
                # print(output.shape)
                # print(decoder_output.numpy().shape)
                if output is None:
                    output = decoder_output
                else:
                    output = tf.concat([output, decoder_output], axis=1)
                state = decoder_hidden
                if i + 1 in range(Dialog.max_dialog_len):
                    next_input = tf.slice(inputs, begin=[0, i+1], size=[-1, 1])
                else:
                    break
            output = tf.nn.softmax(output, 2)
        return output
