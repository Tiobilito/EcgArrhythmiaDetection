import argparse
from keras.models import load_model
from data_processing import prepare_datasets
from utils import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True,
                        help="Ruta al .h5 que se desea evaluar")
    args = parser.parse_args()

    _, _, (X_test, y_test), _ = prepare_datasets()
    model = load_model(args.model_path)
    model_name = args.model_path.split('/')[-1].split('.')[0]
    evaluate_model(model, X_test, y_test, model_name)

if __name__ == '__main__':
    main()
