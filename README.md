# ActiveLearning for Segmantation with UncertaintyIndicators in PyTorch
ActiveLearning with Uncertainty Indicators for SemanticSegmantation


### Prerequisites:
- Linux or macOS
- Python 3.8
- CPU compatible but NVIDIA GPU + CUDA CuDNN is highly recommended.

### Experiments and Visualization
Please edit --data_dir in auguments.py to where your Cityscapes data is.


The code can simply be run using 
```
python3 main.py
```
If you want to use GPU
```
python3 main.py --cuda
```

The results will be saved in `results/accuracies.log`.
