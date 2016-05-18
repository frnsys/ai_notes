from: <http://danijar.github.io/introduction-to-recurrent-networks-in-tensorflow>
by Danijar Hafner

# Introduction to Recurrent Networks in TensorFlow

May 5, 2016

Recurrent networks like LSTM and GRU are powerful sequence models. I will explain how to create recurrent networks in TensorFlow and use them for sequence classification and sequence labelling tasks. If you are not familiar with recurrent networks, I suggest you take a look at Christopher Olah's [great post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) first. On the TensorFlow part, I also expect some basic knowledge. The [official tutorials](https://www.tensorflow.org/versions/r0.8/tutorials/index.html) are a good place to start.

## Defining the Network

To use recurrent networks in TensorFlow we first need to define the network architecture consiting of one or more layers, the cell type and possibly dropout between the layers.

    from tensorflow.models.rnn import rnn_cell

    num_hidden = 200
    num_layers = 3
    dropout = tf.placeholder(tf.float32)

    network = rnn_cell.GRUCell(num_hidden)  # Or LSTMCell(num_hidden)
    network = rnn_cell.DropoutWrapper(network, output_keep_prob=dropout)
    network = rnn_cell.MultiRNNCell([network] * num_layers)

## Unrolling in Time

We can now unroll this network in time using the `rnn` operation. This takes placeholders for the input at each timestep and returns the hidden _states_ and _output_ activations for each timestep.

    from tensorflow.models.rnn import rnn
    max_length = 100

    # Batch size times time steps times data width.
    data = tf.placeholder(tf.float32, [None, max_length, 28])
    outputs, states = rnn.rnn(network, unpack_sequence(data))
    output = pack_sequence(outputs)
    state = pack_sequence(states)

TensorFlow uses Python lists of one tensor for each timestep for the interface. Thus we make use of [`tf.pack()` and `tf.unpack()`](https://www.tensorflow.org/versions/r0.8/api_docs/python/array_ops.html#pack) to split our data tensors into lists of frames and merge the results back to a single tensor.

    def unpack_sequence(tensor):
        """Split the single tensor of a sequence into a list of frames."""
        return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

    def pack_sequence(sequence):
        """Combine a list of the frames into a single tensor of the sequence."""
        return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

## Sequence Classification

For classification, you might only care about the output activation at the last timestep, which is just `outputs[-1]`. For now we assume sequences to be equal in length but I will cover variable length sequences in another post.

    in_size = num_hidden
    out_size = int(target.get_shape()[2])
    weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
    prediction = tf.nn.softmax(tf.matmul(outputs[-1], weight) + bias)

The code just adds a softmax layer ontop of the recurrent network that tries to predict the target from the last RNN activation.

    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

## Sequence Labelling

For sequence labelling, we want a prediction for each timestamp. However, we share the weights for the softmax layer across all timesteps. This way, we have one softmax layer ontop of an unrolled recurrent network as desired.

    in_size = num_hidden
    out_size = int(target.get_shape()[2])
    weight = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
    predictions = [tf.nn.softmax(tf.matmul(x, weight) + bias) for x in outputs]
    prediction = pack_sequence(predictions)

Since this also is a classification task, we keep using cross entropy. We first compute the cross entropy for every timestep and then average.

    cross_entropy = -tf.reduce_sum(
        target * tf.log(prediction), reduction_indices=[1])
    cross_entropy = tf.reduce_mean(cross_entropy)

We learned how to construct recurrent networks in TensorFlow and use them for sequence learning tasks. Please ask any questions below if you couldn't follow.