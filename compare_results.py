import os
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def main():
    summary_csv = os.path.join(RESULTS_DIR, "results_summary.csv")
    df = pd.read_csv(summary_csv)

    # Comparación de accuracy
    plt.figure()
    plt.bar(df['model'], df['val_accuracy'])
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Validation Accuracy")
    plt.savefig(os.path.join(RESULTS_DIR, "val_accuracy_comparison.png"))
    plt.close()
    print(f"Gráfica de comparación guardada en {RESULTS_DIR}/val_accuracy_comparison.png")

    # Mostrar la tabla resumen
    print(df)

if __name__ == '__main__':
    main()
