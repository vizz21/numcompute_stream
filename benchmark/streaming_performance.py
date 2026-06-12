import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numcompute_stream import (
    StreamingDecisionTreeClassifier,
    RandomForestClassifier,
    Pipeline,
    StandardScaler,
    load_csv,
    StreamTrainer
)

def run_accuracy_benchmark():
    """
    Functino to run accuracy benchmark
    """
    data = load_csv(os.path.join(os.path.dirname(__file__), '../demo/iris.csv'), encode_labels=True)

    X = data[:, :-1]
    y = data[:, -1]
    
    sorted_indices = np.argsort(y)
    X, y = X[sorted_indices], y[sorted_indices]
    
    n_chunks = 10
    chunk_size = X.shape[0] // n_chunks
    
    trainer_dt = StreamTrainer(Pipeline([
        ('s', StandardScaler()), 
        ('m', StreamingDecisionTreeClassifier(max_depth=3))
    ]))
    
    trainer_rf = StreamTrainer(Pipeline([
        ('s', StandardScaler()), 
        ('m', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42))
    ]))
    
    dt_accs, rf_accs = [], []

    for i in range(n_chunks):
        start, end = i * chunk_size, (i + 1) * chunk_size
        Xc, yc = X[start:end], y[start:end]
        try:
            dt_acc = trainer_dt.score_chunk(Xc, yc)['accuracy']
            rf_acc = trainer_rf.score_chunk(Xc, yc)['accuracy']
        except ValueError:
            dt_acc = 0.0
            rf_acc = 0.0
    
        dt_accs.append(dt_acc)
        rf_accs.append(rf_acc)

        trainer_dt.update(Xc, yc)
        trainer_rf.update(Xc, yc)
    
    y_pred_dt_all = trainer_dt.pipeline.predict(X)
    y_pred_rf_all = trainer_rf.pipeline.predict(X)
    
    final_dt = np.mean(y_pred_dt_all == y)
    final_rf = np.mean(y_pred_rf_all == y)
    initial_dt = dt_accs[0]
    initial_rf = rf_accs[0]

    print(f"{'Metric':<25} | {'Single DT':<15} | {'Random Forest ':<15}")
    print("-" * 65)
    print(f"{'Initial Accuracy':<25} | {initial_dt:<15.4f} | {initial_rf:<15.4f}")
    print(f"{'Final Cumulative Accuracy':<25} | {final_dt:<15.4f} | {final_rf:<15.4f}")

if __name__ == "__main__":
    run_accuracy_benchmark()