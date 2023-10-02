import tensorflow as tf
tf.config.run_functions_eagerly(True)



class NeuralNetworkTf(tf.keras.Sequential):

  def __init__(self, sizes, random_state=1):
    
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)
    
    # Adding input layer (The model doesn't specify an input layer to determine the shape of the input data)
    self.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    for i in range(0, len(sizes)- 1):
      # We use sigmoid activations for the hidden layers and a softmax activation only for the output layer for multi-class classification problems. Here I switched them up
      # Because softmax function will output the vector with probabilities and the sum of all probabilities corresponding to every class is 1
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid'))
    self.add(tf.keras.layers.Dense(sizes[-1], activation='softmax'))
        
  
  def compile_and_fit(self, x_train, y_train, 
                      epochs=50, learning_rate=0.01, 
                      batch_size=1,validation_data=None):
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # For multi-class classification, we should use 'CategoricalCrossentropy' loss instead of 'BinaryCrossentropy'. It is also indicated in the Data part
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    eval_metrics = ['accuracy']

    super().compile(optimizer=optimizer, loss=loss_function, 
                    metrics=eval_metrics)
    return super().fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=validation_data)  



class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    super().__init__()
    # Here we store the initial_learning_rate as TensorFlow variable, and the variable should not be updated during training
    self.initial_learning_rate = tf.Variable(initial_learning_rate, trainable=False, dtype=tf.float32)



  def __call__(self, step):
    # Here we need to ensure that the learning rate does not fall below 1. The function will return the larger value between (self.initial_learning_rate - step) and 1
    # So, when the computed learning rate falls below 1, it will return 1 instead
    decayed_learning_rate = tf.math.maximum(self.initial_learning_rate - step, 1.0)
    return decayed_learning_rate
  