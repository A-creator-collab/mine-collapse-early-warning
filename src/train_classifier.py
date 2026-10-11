"""Train 3-class classifier using scikit-learn."""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import yaml


class ClassifierTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.model_cfg = self.config['model']['classifier']
    
    def build_model(self):
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.model_cfg['hidden_layers']),
            activation='relu',
            solver='adam',
            learning_rate_init=self.model_cfg['learning_rate'],
            batch_size=self.model_cfg['batch_size'],
            max_iter=self.model_cfg['epochs'] * 10,
            early_stopping=True,
            n_iter_no_change=self.model_cfg['early_stopping_patience'],
            validation_fraction=0.1,
            random_state=42,
            verbose=True
        )
    
    def train(self):
        print("Loading preprocessed data...")
        X_train = np.load('data/processed/X_train.npy')
        X_val = np.load('data/processed/X_val.npy')
        X_test = np.load('data/processed/X_test.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_val = np.load('data/processed/y_val.npy')
        y_test = np.load('data/processed/y_test.npy')

        print(f"Train label distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")

        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        model = self.build_model()
        print("\nTraining MLPClassifier...")
        model.fit(X_train_full, y_train_full)

        print("\n=== Test Set Evaluation ===")
        y_pred = model.predict(X_test)
        print(classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=['Normal', 'Warning', 'Critical'],
            zero_division=0
        ))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Warning', 'Critical'],
                    yticklabels=['Normal', 'Warning', 'Critical'])
        plt.title('Classifier Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        Path("models").mkdir(exist_ok=True)
        plt.savefig('models/confusion_matrix.png', dpi=100)
        plt.close()

        joblib.dump(model, self.config['paths']['classifier_model'].replace('.keras', '.pkl'))
        print("Model saved.")
        return model

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    ClassifierTrainer().train()
    print("Classifier training complete.")