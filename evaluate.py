import os
import argparse
import pandas as pd

from sklearn.metrics import classification_report
from keras.models import load_model

from data_processing import prepare_datasets
from utils import evaluate_model
from config import RESULTS_DIR, LABELS

def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo entrenado y vuelca resultados en su carpeta de run"
    )
    parser.add_argument('--model-path', required=True,
                        help="Ruta a models/saved/<run_name>.h5")
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    # Extrae run_name y prepara carpeta
    run_name   = os.path.basename(model_path).replace('.h5', '')
    run_folder = os.path.join(RESULTS_DIR, run_name)
    plots_dir  = os.path.join(run_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Carga datos
    _, _, (X_test, y_test), _ = prepare_datasets()

    # Carga modelo
    model = load_model(model_path)

    # 1) Evalúa loss y accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    df_tm = pd.DataFrame([{
        'test_loss'    : loss,
        'test_accuracy': accuracy
    }])
    df_tm.to_csv(os.path.join(run_folder, "test_metrics.csv"), index=False)

    # 2) Guarda clasificación detallada
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    cr_dict = classification_report(
        y_true, y_pred,
        target_names=list(LABELS.values()),
        output_dict=True
    )
    cr_df = pd.DataFrame(cr_dict).transpose()
    cr_df.to_csv(os.path.join(run_folder, "classification_report.csv"), index=True)

    # 3) Guarda matriz de confusión
    evaluate_model(model, X_test, y_test, run_name, plots_dir)

    print(f"\n✅ Evaluate completo. Resultados en: {run_folder}")

if __name__ == '__main__':
    main()
