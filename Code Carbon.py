from codecarbon import EmissionsTracker 

tracker = EmissionsTracker() 
tracker.start() 

# Example: Training a Machine Learning model 
import tensorflow as tf 
model = tf.keras.Sequential([ 
 tf.keras.layers.Dense(64, activation='relu'), 
 tf.keras.layers.Dense(1) 
]) 
model.compile(optimizer='adam', loss='mean_squared_error') 
model.fit(x_train, y_train, epochs=10) 

tracker.stop() 

