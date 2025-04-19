import os
import argparse
import csv

from data_processing import prepare_datasets
from models.ann import build_ann
from models.cnn import build_cnn
from models.rnn import build_rnn
from utils import plot_history
from config import MODELS_SAVED_DIR, RESULTS_DIR, HYPERPARAMS

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Asegurarse de que existan estas carpetas
os.makedirs(MODELS_SAVED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def make_run_name(args):
    # base con tipo modelo, epochs, batch_size, learning_rate
    lr_str = f"{args.learning_rate:.0e}"
    parts = [args.model, f"e{args.epochs}", f"bs{args.batch_size}", f"lr{lr_str}"]
    # anexa estructura específica
    if args.model == 'ann':
        sizes = "-".join(str(h) for h in args.ann_hidden_sizes)
        parts.append(f"h{sizes}")
    if args.model == 'cnn':
        flts = "-".join(str(f) for f in args.cnn_filters)
        parts.append(f"f{flts}")
    if args.model == 'rnn':
        units = "-".join(str(u) for u in args.rnn_units)
        parts.append(f"u{units}")
    # dropout (reemplazar punto por 'p' para filesystem)
    drs = "-".join(str(dr).replace('.', 'p') for dr in args.dropout_rates)
    parts.append(f"d{drs}")
    return "_".join(parts)

def get_model(name, input_shape, kw):
    if name == 'ann':
        return build_ann(input_shape, kw['hidden_sizes'], kw['dropout_rates'])
    if name == 'cnn':
        return build_cnn(input_shape, kw['filters'], kw['dropout_rates'])
    if name == 'rnn':
        return build_rnn(input_shape, kw['units'], kw['dropout_rates'])
    raise ValueError('Modelo no reconocido')

def main():
    parser = argparse.ArgumentParser(description="Entrena ANN/CNN/RNN sobre ECG MIT-BIH")
    parser.add_argument('--model',           choices=['ann','cnn','rnn'], required=True)
    parser.add_argument('--epochs',          type=int,   default=HYPERPARAMS['epochs'])
    parser.add_argument('--batch_size',      type=int,   default=HYPERPARAMS['batch_size'])
    parser.add_argument('--learning_rate',   type=float, default=HYPERPARAMS['learning_rate'])
    parser.add_argument('--test_size',       type=float, default=HYPERPARAMS['test_size'])
    parser.add_argument('--oversample',      type=lambda x: x.lower()=='true', default=HYPERPARAMS['oversample'])

    parser.add_argument('--ann_hidden_sizes', nargs='+', type=int,   default=[128,64])
    parser.add_argument('--cnn_filters',      nargs='+', type=int,   default=[32,64,128])
    parser.add_argument('--rnn_units',        nargs='+', type=int,   default=[64,64])
    parser.add_argument('--dropout_rates',    nargs='+', type=float, default=[0.5,0.5])

    args = parser.parse_args()

    # Actualizar globals
    HYPERPARAMS.update({
        'epochs'        : args.epochs,
        'batch_size'    : args.batch_size,
        'learning_rate' : args.learning_rate,
        'test_size'     : args.test_size,
        'oversample'    : args.oversample
    })

    # Preparar datos
    (X_train, y_train), (X_val, y_val), _, _ = prepare_datasets()
    input_shape = X_train.shape[1:]

    # Construir modelo
    model_kw = {}
    if args.model == 'ann':
        model_kw = {'hidden_sizes': args.ann_hidden_sizes, 'dropout_rates': args.dropout_rates}
    if args.model == 'cnn':
        model_kw = {'filters': args.cnn_filters, 'dropout_rates': args.dropout_rates}
    if args.model == 'rnn':
        model_kw = {'units': args.rnn_units, 'dropout_rates': args.dropout_rates}

    model = get_model(args.model, input_shape, model_kw)

    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3),
        # guardamos modelo "best" en carpeta temporal, pero renombraremos al final
        ModelCheckpoint("temp_best.h5", save_best_only=True)
    ]

    # ================================================
    # 1) Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=HYPERPARAMS['epochs'],
        batch_size=HYPERPARAMS['batch_size'],
        callbacks=callbacks
    )

    # 2) Nombre y carpetas por run
    run_name       = make_run_name(args)
    run_folder     = os.path.join(RESULTS_DIR, run_name)
    plots_folder   = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # 3) Guardar curvas
    plot_history(history, run_name, plots_folder)

    # 4) Guardar modelo final y best
    os.replace("temp_best.h5",
               os.path.join(MODELS_SAVED_DIR, f"{run_name}.h5"))
    model.save(os.path.join(MODELS_SAVED_DIR, f"{run_name}_final.h5"))

    # 5) Guardar métricas e hiper‑parámetros
    val_acc  = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]

    metrics_file = os.path.join(run_folder, "metrics.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model","epochs","batch_size","learning_rate","test_size","oversample",
            "hidden_sizes","filters","units","dropout_rates","val_accuracy","val_loss"
        ])
        writer.writerow([
            args.model,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.test_size,
            args.oversample,
            ",".join(map(str, args.ann_hidden_sizes)) if args.model=="ann" else "",
            ",".join(map(str, args.cnn_filters))      if args.model=="cnn" else "",
            ",".join(map(str, args.rnn_units))        if args.model=="rnn" else "",
            ",".join(map(str, args.dropout_rates)),
            val_acc,
            val_loss
        ])

    print(f"Resultados guardados en: {run_folder}")
    print(f"Modelos en: {MODELS_SAVED_DIR}")

if __name__ == '__main__':
    main()
