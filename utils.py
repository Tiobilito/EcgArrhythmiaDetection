import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from config import LABELS

def plot_history(history, model_name, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_accuracy.png"))
    plt.close()
    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_loss.png"))
    plt.close()

def evaluate_model(model, X_test, y_test, model_name, plots_dir=None):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Reporte en consola
    print(classification_report(y_true, y_pred, target_names=list(LABELS.values())))

    # Matriz de confusión
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        plt.imshow(cm, cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(range(len(LABELS)), list(LABELS.values()), rotation=45)
        plt.yticks(range(len(LABELS)), list(LABELS.values()))
        for i in range(len(cm)):
            for j in range(len(cm)):
                plt.text(j, i, cm[i,j], ha='center', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
