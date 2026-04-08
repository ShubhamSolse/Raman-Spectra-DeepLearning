# Raman Spectra Deep Learning

Deep learning models for Raman spectra classification using PyTorch.

## What is Raman Spectra?

Raman spectroscopy is a non-destructive analytical technique that measures the interaction of light with molecular vibrations. When light interacts with a sample, most photons scatter elastically (Rayleigh scattering), but a small fraction scatter inelastically (Raman scattering), shifting in wavelength. This shift provides a unique molecular fingerprint that can identify chemical composition and structure.

**Applications:**
- Material characterization
- Pharmaceutical analysis
- Biomedical diagnostics
- Quality control
- Chemical identification

This project uses deep learning to classify Raman spectra patterns for automated analysis.

## Models

- **CNN**: Convolutional Neural Network with 1D convolutions
- **LSTM**: Recurrent network with stacked LSTM layers
- **Transformer**: LSTM + multi-head self-attention
- **GCN**: Graph Convolutional Network
- **CLR**: Contrastive Learning with CNN backbone

## Requirements

```bash
pip install torch pandas scikit-learn matplotlib streamlit
```

## Usage

### Command Line

```bash
python main.py --model CLR --epochs 100 --lr 0.01 --hidden 64 --c 2
```

**Arguments:**
- `--use-cuda`: Enable GPU training
- `--seed`: Random seed (default: 1)
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 0.01)
- `--wd`: Weight decay (default: 1e-5)
- `--hidden`: Hidden dimension (default: 64)
- `--c`: Number of classes - 2 for binary, >2 for multi-class (default: 2)
- `--d`: Spectra dimension (default: 1200)
- `--model`: Model type - CNN/LSTM/Transformer/CLR (default: CLR)

### Streamlit Interface

```bash
streamlit run streamlit.py
```

Interactive web interface for model selection and hyperparameter tuning.

## Data

- `bin.csv`: Binary classification dataset
- `multi.csv`: Multi-class classification dataset

Format: First 1200 columns are spectra features, last column is label.

## Evaluation

5-fold cross-validation with metrics:
- **Binary**: AUC, Sensitivity, Specificity, Precision, Accuracy, F1, MCC
- **Multi-class**: Recall, Precision, Accuracy, F1

## Files

- `main.py`: Training script with argparse
- `streamlit.py`: Web interface
- `transformer.py`: Transformer block implementation
- `utils.py`: Helper functions and metrics
