# CGT registry and predictions

Small check that a coloured graph tensor keeps features, labels, and colours apart, wraps `run_ds` node predictions, reads a unitig sequence from the source CDBG, and derives CSC without reordering edge features.

```bash
python examples/cgt_ml/run.py
```

The script does not report model accuracy. For these softmax models, confidence is the predicted class probability.
