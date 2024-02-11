#1
base_mdl = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_mdl.trainable = False
sign_lang_model = keras.models.Sequential()
sign_lang_model.add(base_mdl)
sign_lang_model.add(keras.layers.Flatten()) 
sign_lang_model.add(keras.layers.Dense(256, activation=tf.nn.relu))
sign_lang_model.add(keras.layers.Dropout(.3))
sign_lang_model.add(keras.layers.Dense(36, activation=tf.nn.softmax))
sign_lang_model.summary()
