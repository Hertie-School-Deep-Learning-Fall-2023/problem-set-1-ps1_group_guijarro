import tensorflow as tf
tf.config.run_functions_eagerly(True)



class NeuralNetworkTf(tf.keras.Sequential):

  def __init__(self, sizes, random_state=1):
    
    super().__init__()
    self.sizes = sizes
    self.random_state = random_state
    tf.random.set_seed(random_state)
    
    for i in range(0, len(sizes)):

      if i == len(sizes) - 1:
        self.add(tf.keras.layers.Dense(sizes[i], activation='sigmoid'))
        
      else:
        self.add(tf.keras.layers.Dense(sizes[i], activation='softmax'))
        
  
  def compile_and_fit(self, x_train, y_train, 
                      epochs=50, learning_rate=0.01, 
                      batch_size=1,validation_data=None):
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_function = tf.keras.losses.BinaryCrossentropy()
    eval_metrics = ['accuracy']

    super().compile(optimizer=optimizer, loss=loss_function, 
                    metrics=eval_metrics)
    return super().fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=validation_data)  



class TimeBasedLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''TODO: Implement a time-based learning rate that takes as input a 
  positive integer (initial_learning_rate) and at each step reduces the
  learning rate by 1 until minimal learning rate of 1 is reached.
    '''

  def __init__(self, initial_learning_rate):

    pass


  def __call__(self, step):

    pass


    