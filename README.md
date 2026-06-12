# NumCompute Stream:
NumCompute Stream is built entirely using **NumPy**.
It can be used processing data incrementally, handling concept drift and visualisation  without relying on heavy external libraries like Scikit-Learn



## Key Features
- `StreamingDecisionTreeClassifier`: A depth-limited decision tree
- `RandomForestClassifier`: A ensemble method supporting incremental updates
- `StandardScaler`: Incremental mean and variance calculation
- `SimpleImputer`: Streaming-compatible missing value handling
- `Pipeline`: Modular chaining of preprocessing and model steps with full `.partial_fit()` 
- **Prequential Evaluation**: Integrated `StreamTrainer` for "Test-Then-Train" evaluation
- **Real-time Visualization**:
  - Monitoring accuracy and metrics over time
  - Model comparison and performance diagnostics
  - Interactive confusion matrices and prediction analysis
- **Numerical Stability**: Optimized for precision and memory safety in long-running streaming environments


## Installation & Requirements

The framework requires only standard Python data science libraries:
- Python 3.8+
- NumPy
- Matplotlib


```bash
git clone https://github.com/vizz21/numcompute_stream.git
cd numcompute_stream
```


## Quick Start: Prequential Evaluation

```python
from numcompute_stream import (
    Pipeline, StandardScaler, StreamingDecisionTreeClassifier, StreamTrainer
)

# Initialize a streaming pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', StreamingDecisionTreeClassifier(max_depth=5))
])

# Initialize the trainer
trainer = StreamTrainer(pipeline)

# Simulate streaming data chunks
for X_chunk, y_chunk in data_stream:
    # Test on unseen data first, then train
    metrics = trainer.score_chunk(X_chunk, y_chunk)
    trainer.update(X_chunk, y_chunk)
```
## Testing

```bash
python -m unittest discover tests
```


## Demo
Demo is provided in demo/stream_demo.ipynb