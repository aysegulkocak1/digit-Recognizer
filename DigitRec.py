from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DigitRecognizer:
    def __init__(self) :
        self.EPOCHS = 10
        self.TEST_SIZE = 0.33
        self.RANDOM_STATE = 42
        self.digits = load_digits()
        self.NUM_CATEGORIES = 10  

    def get_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((self.digits.images.shape[1], self.digits.images.shape[2], 1), input_shape=(self.digits.images.shape[1], self.digits.images.shape[2])),
            tf.keras.layers.Conv2D(64, (3, 3), activation="swish"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='swish'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.NUM_CATEGORIES, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",  
            metrics=["accuracy"]
        )
        return model

    def recognize(self):
        x_train, x_test, y_train, y_test = train_test_split(self.digits.images, self.digits.target, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE)
        
        model = self.get_model()
        model.summary()  
        
        history = model.fit(x_train, y_train, epochs=self.EPOCHS, validation_data=(x_test, y_test))
        print("Train score:", model.evaluate(x_train, y_train))
        print("Test score:", model.evaluate(x_test, y_test))
        self.plotGraphics(history)
        
        model.save("digitmodels.h5")  


    def plotGraphics(self,history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()




recognizer = DigitRecognizer()
recognizer.recognize()
