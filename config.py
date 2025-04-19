import os

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))

DATA_DIR           = os.path.join(BASE_DIR, "DATASET-ECG")
TRAIN_FILE         = os.path.join(DATA_DIR, "mitbih_train.csv")
TEST_FILE          = os.path.join(DATA_DIR, "mitbih_test.csv")

MODELS_SAVED_DIR   = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR        = os.path.join(BASE_DIR, "results")

LABELS = {
    0: "Normal",
    1: "Artial Premature",
    2: "Premature ventricular contraction",
    3: "Fusion of ventricular and normal",
    4: "Fusion of paced and normal"
}

HYPERPARAMS = {
    'batch_size'    : 64,
    'epochs'        : 50,
    'learning_rate' : 1e-3,
    'test_size'     : 0.2,
    'random_state'  : 42,
    'oversample'    : True
}
