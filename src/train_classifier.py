"""Train 3-class classifier."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml


class ClassifierTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_cfg = self.config['model']['classifier']
    
    def build_model(self, input_dim: int, n_classes: int = 3) -> keras.Model:
        layers_list = [layers.Input(shape=(input_dim,))]
        for i, units in enumerate(self.model_cfg['hidden_layers']):
            layers_list.append(layers.Dense(units, activation='relu'))
            layers_list.append(layers.BatchNormalization())
            if i < len(self.model_cfg['hidden_layers']) - 1:
                layers_list.append(layers.Dropout(self.model_cfg['dropout']))
        layers_list.append(layers.Dense(n_classes, activation='softmax'))
        model = keras.Sequential(layers_list)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.model_cfg['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model
    
    def train(self):
        print("Loading preprocessed data...")
        X_train = np.load('data/processed/X_train.npy')
        X_val = np.load('data/processed/X_val.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_val = np.load('data/processed/y_val.npy')
        y_test = np.load('data/processed/y_test.npy')
        model = self.build_model(X_train.shape[1])
        model.summary()
        cbs = [
            callbacks.EarlyStopping(monitor='val_loss',
                patience=self.model_cfg['early_stopping_patience'],
                restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                patience=3, min_lr=1e-6),
            callbacks.ModelCheckpoint(self.config['paths']['classifier_model'],
                monitor='val_accuracy', save_best_only=True)
        ]
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=self.model_cfg['epochs'], batch_size=self.model_cfg['batch_size'],
            callbacks=cbs, verbose=1)
        print("\n=== Test Set Evaluation ===")
        y_pred = model.predict(X_test, verbose=0)
        y_pred_cls = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred_cls,
            target_names=['Normal', 'Warning', 'Critical']))
        cm = confusion_matrix(y_test, y_pred_cls)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Warning', 'Critical'],
                    yticklabels=['Normal', 'Warning', 'Critical'])
        plt.title('Classifier Confusion Matrix')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=100)
        plt.close()
        return model


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    ClassifierTrainer().train()
    print("Classifier training complete.")