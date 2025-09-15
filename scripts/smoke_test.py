import os, sys
CURRENT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
from drought_app.utils.sample import make_synthetic_dataset
from drought_app.utils.nddi import compute_nddi_column
from drought_app.core.model import SeriesWindow, build_lstm_model, train_model, mc_dropout_predict


def main():
    df = make_synthetic_dataset(n=180)
    df = compute_nddi_column(df, 'NDVI', 'NDWI', 'NDDI')
    df = df.set_index('date').resample('D').mean()
    nddi = df['NDDI'].astype(float).values

    sw = SeriesWindow(nddi, lookback=12, horizon=1)
    X_train, y_train, X_val, y_val = sw.get_train_val_split()
    model = build_lstm_model(input_steps=12, units=16, dropout=0.2)
    train_model(model, X_train, y_train, X_val, y_val, epochs=2, batch_size=32)

    preds = mc_dropout_predict(model, sw.get_last_window(), mc_samples=10)
    print('Smoke OK: pred mean/std', float(np.mean(preds)), float(np.std(preds)))


if __name__ == '__main__':
    main()
