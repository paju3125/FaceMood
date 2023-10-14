import tensorflow as tf

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))


def my_new_model():
    img_inputs = tf.keras.Input(shape=(48, 48, 3))

    # base model
    x = base_model(img_inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = tf.keras.layers.Dense(1024, activation="relu", name='d1')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(1024, activation="relu", name='d2')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output = tf.keras.layers.Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=img_inputs, outputs=output, name='Face_Emotion_Model_With_VGG')

    model.load_weights('./models/new_model/checkpoint (1)')

    return model
