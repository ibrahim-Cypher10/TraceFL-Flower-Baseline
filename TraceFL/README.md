---
title: TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance
url: https://arxiv.org/abs/2312.13632
labels: [interpretability, federated-learning, neuron-provenance, debugging]
dataset: [MNIST, CIFAR-10, PathMNIST, YahooAnswers]
---

[![Flower](https://img.shields.io/badge/Flower-Federated%20Learning-blue)](https://flower.dev)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

> [!NOTE]
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [https://arxiv.org/abs/2312.13632](https://arxiv.org/abs/2312.13632)

## What's Implemented

This baseline implements TraceFL, a framework for interpretability-driven debugging in federated learning via neuron provenance. The implementation includes:

1. Correct Prediction Tracing (Figure 2)
2. Neuron Provenance Analysis (Figure 3)
3. Client Attribution Analysis (Figure 5)
4. Performance Metrics (Table 3)

## Running the Experiments

### Experiment 1 — Correct Prediction Tracing (Figure 2)
This experiment demonstrates how neuron activations trace back to client contributions for correctly predicted samples.

```bash
flwr run . --run-config exp_1
```

Expected Results:
- Figure 2: Shows the tracing of neuron activations to client contributions
- Results will be saved in `results/exp_1/`
- Use `python visualize_results.py --exp exp_1 --figure 2` to generate the figure

### Experiment 2 — Neuron Provenance Analysis (Figure 3)
This experiment analyzes the distribution of neuron activations across clients.

```bash
flwr run . --run-config exp_2
```

Expected Results:
- Figure 3: Shows neuron activation distributions and their contribution to predictions
- Results will be saved in `results/exp_2/`
- Use `python visualize_results.py --exp exp_2 --figure 3` to generate the figure

### Experiment 3 — Client Attribution Analysis (Figure 5)
This experiment analyzes how different clients contribute to model performance.

```bash
flwr run . --run-config exp_3
```

Expected Results:
- Figure 5: Shows client contributions to model performance across classes
- Results will be saved in `results/exp_3/`
- Use `python visualize_results.py --exp exp_3 --figure 5` to generate the figure

### Performance Metrics (Table 3)
To generate the performance metrics table:

```bash
python visualize_results.py --table 3
```

This will generate Table 3 showing performance metrics across all experiments.

## Visualization and Analysis

The `visualize_results.py` script provides tools to:
1. Load experiment results
2. Generate figures from the paper
3. Create performance metrics tables

Example usage:
```bash
# Generate a specific figure
python visualize_results.py --exp exp_1 --figure 2

# Generate the performance table
python visualize_results.py --table 3

# Generate all figures and tables
python visualize_results.py --all
```

## Project Structure

```
tracefl/
├── config/                 # Experiment configurations
├── tracefl/               # Core implementation
│   ├── client_app.py      # Client implementation
│   ├── server_app.py      # Server implementation
│   ├── strategy.py        # FL strategy
│   └── ...
├── visualize_results.py   # Visualization tools
└── README.md             # This file
```

## Dependencies

See `pyproject.toml` for a complete list of dependencies.

## Citation

If you use this baseline in your work, please cite:

```bibtex
@article{tracefl2023,
  title={TraceFL: Interpretability-Driven Debugging in Federated Learning via Neuron Provenance},
  author={...},
  journal={arXiv preprint arXiv:2312.13632},
  year={2023}
}
```

---

> Maintained as part of the Flower Baselines Collection
