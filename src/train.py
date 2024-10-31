import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Build a Keras model given some parameters
def create_model(
    captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2
):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for _ in range(module_length):
            x = keras.layers.Conv2D(32 * 2 ** min(i, 3),kernel_size=3,padding="same",kernel_initializer="he_uniform")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = [keras.layers.Dense(captcha_num_symbols, activation="softmax", name="char_%d" % (i + 1))(x) for i in range(captcha_length)]
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model


# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(
        self,
        directory_name,
        batch_size,
        captcha_length,
        captcha_symbols,
        captcha_width,
        captcha_height,
    ):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split(".")[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros(
            (self.batch_size, self.captcha_height, self.captcha_width, 3),
            dtype=numpy.float32,
        )
        y = [
            numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8)
            for i in range(self.captcha_length)
        ]

        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            X[i] = processed_data

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.

            random_image_label = random_image_label.split("_")[0]

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        self.files.update(
            dict(zip(map(lambda x: x.split(".")[0], self.used_files), self.used_files))
        )
        self.used_files = []
        return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--width", help="Width of captcha image", type=int,required=True)
    parser.add_argument("-H", "--height", help="Height of captcha image", type=int,required=True)
    parser.add_argument("-l", "--length", help="Length of captchas in characters", type=int,required=True)
    parser.add_argument("-b", "--batch-size", help="How many images in training captcha batches", type=int,required=True)
    parser.add_argument("-t", "--train-dataset", help="Where to look for the training image dataset", type=str,required=True)
    parser.add_argument("-v", "--validate-dataset",help="Where to look for the validation image dataset",type=str,required=True)
    parser.add_argument("-o", "--output-model-name", help="Where to save the trained model", type=str,required=True)
    parser.add_argument("-r", "--input-model",help="Where to look for the input model to continue training",type=str,required=False)
    parser.add_argument("-e", "--epochs", help="How many training epochs to run", type=int,required=True)
    parser.add_argument("-s", "--symbols", help="File with the symbols to use in captchas", type=str,required=True)
    parser.add_argument("-p", "--is-tflite", help="Set model to export .tflite", action='store_true',default=False)
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_model_name)):
        os.makedirs(os.path.dirname(args.output_model_name))
    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "No GPU available!"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # with tf.device('/device:GPU:0'):
    with tf.device("/cpu:0"):
        # with tf.device('/device:XLA_CPU:0'):
        model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 3))

        if args.input_model is not None:
            model.load_weights(args.input_model)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            metrics=["accuracy"],
        )

        model.summary()

        training_data = ImageSequence(
            args.train_dataset,
            args.batch_size,
            args.length,
            captcha_symbols,
            args.width,
            args.height,
        )
        validation_data = ImageSequence(
            args.validate_dataset,
            args.batch_size,
            args.length,
            captcha_symbols,
            args.width,
            args.height,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=args.epochs // 10, restore_best_weights=True
            ),
            # keras.callbacks.CSVLogger('log.csv'),
            keras.callbacks.ModelCheckpoint(
                args.output_model_name + ".h5", save_best_only=True
            ),
        ]

        # Save the model architecture to JSON
        with open(args.output_model_name + ".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit(
                x=training_data,
                validation_data=validation_data,
                epochs=args.epochs,
                callbacks=callbacks,
                use_multiprocessing=True,
            )
        except KeyboardInterrupt:
            print(
                "KeyboardInterrupt caught, saving current weights as "
                + args.output_model_name
                + "_resume.h5"
            )
            model.save_weights(args.output_model_name + "_resume.h5")
        finally:
            if args.is_tflite:
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                lite_model = converter.convert()
                with open(f'{args.output_model_name}.tflite', 'wb') as lite_file:
                    lite_file.write(lite_model)


if __name__ == "__main__":
    main()
