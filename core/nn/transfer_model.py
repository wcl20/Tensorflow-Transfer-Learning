from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model

class TransferModel:

    @staticmethod
    def build(base_model, D, classes):

        # Make a head for base model output
        head = base_model.output
        head = Flatten()(head)
        head = Dense(D, activation="relu")(head)
        head = Dropout(0.5)(head)
        head = Dense(classes, activation="softmax")(head)

        # Combine base with head 
        model = Model(inputs=base_model.input, outputs=head)
        return model
