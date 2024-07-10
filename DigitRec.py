from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

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

        p = model.predict(x_test)
        cm = confusion_matrix(y_test,p.argmax(axis = 1))

        self.plotGraphics(history)
        self.plotConfussion(cm,list(range(10)))
        self.plot_missclassified_examples(p.argmax(axis = 1),y_test,x_test)
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

    def plotConfussion(self,cm, classes):
        
        plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
        plt.title("Confussion Matrix")
        plt.colorbar()
        plt.xticks(np.arange(len(classes)),classes,rotation=45)
        plt.yticks(np.arange(len(classes)),classes)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    def plot_missclassified_examples(self,p_test,y_test,x_test):
        misclassified_idx = np.where(p_test != y_test)[0]
        for i in range(len(misclassified_idx)):
            plt.figure()
            plt.imshow(x_test[misclassified_idx[i]], cmap='gray')
            plt.title(f"True label: {y_test[misclassified_idx[i]]} Predicted: {p_test[misclassified_idx[i]]}")
            plt.show()




recognizer = DigitRecognizer()
recognizer.recognize()
