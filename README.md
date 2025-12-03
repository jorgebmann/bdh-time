# BDH Model Implementation

This repository contains the PyTorch implementation of the **BDH (Biological Dragon Hatchling)** model, a novel architecture featuring linear attention with conceptual space projections. The project includes both text classification and financial market prediction applications.

## Project Structure

```
.
├── src/
│   ├── bdh/                    # Core BDH model package
│   │   ├── __init__.py
│   │   ├── model.py           # Core BDH model and attention mechanism
│   │   ├── classifier.py      # Text classification wrapper (for SST-2)
│   │   ├── market.py          # Market prediction model
│   │   └── data.py            # Market dataset loader
│   └── dataset/               # Data ingestion and processing
│       ├── ingestion/         # EODHD API client and parquet storage
│       └── preprocess.py      # Market data preprocessing
├── scripts/
│   ├── train_market.py        # Market model training script
│   ├── build_dataset.py       # Build processed market dataset
│   ├── ingest_stocks.py       # Ingest stock data from EODHD API
│   ├── download_nasdaq100_yfinance.py  # Download NASDAQ100 from yfinance
│   ├── ingest_bulk.py         # Bulk EODHD ingestion (alternative method)
│   └── debug_data.py          # Data debugging utilities (legacy)
├── data/                      # Data storage
│   ├── parquet/               # Parquet-formatted stock data
│   └── market_dataset.pt      # Processed market dataset
├── config.yaml                # Configuration file
├── requirements.txt           # Project dependencies
└── LICENSE                    # License file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bdh-time
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or using `uv` (recommended):
   ```bash
   pip install uv
   uv pip install --system -r requirements.txt
   ```

3. Configure API access (for market data):
   - Copy `config.yaml` and add your EODHD API token:
     ```yaml
     EODHD:
       api_token: your_token_here
     ```

## Usage

### Financial Market Prediction

The market prediction module adapts the BDH model for financial time series, treating assets as particles and learning their interactions to predict future price movements.

#### 1. Data Ingestion

Ingest stock data from the EODHD API:

```bash
python scripts/ingest_stocks.py --start-date 2020-01-01 --end-date 2024-01-01 --exchanges NASDAQ,NYSE
```

Options:
- `--exchanges`: Comma-separated list of exchanges (e.g., "NASDAQ,NYSE")
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD, default: today)
- `--symbols`: Comma-separated list of specific symbols (overrides exchange)
- `--chunk-days`: Days per chunk (default: 90)
- `--resume`: Resume from last checkpoint
- `--library`: Parquet library/directory name (default: stocks)

#### 2. Dataset Generation

Process raw data into optimized tensors:

```bash
python scripts/build_dataset.py
```

This creates `data/market_dataset.pt` with processed features:
- **Positive Momentum** (Excitation): Positive log returns
- **Negative Momentum** (Inhibition): Absolute value of negative log returns
- **Volatility** (Energy): Rolling standard deviation of log returns
- **Normalized Volume** (Activity): Volume normalized by rolling mean

#### 3. Training

Train the MarketBDH model:

```bash
python scripts/train_bidirect_finetune.py
```

The training script includes:
- **Reduced model size** to prevent overfitting (2 layers, 128 embedding dim)
- **Feature normalization** using training statistics
- **Learning rate scheduling** with OneCycleLR
- **Early stopping** based on validation loss
- **Gradient clipping** and weight decay regularization
- **Label smoothing** for better generalization

Training configuration (in `scripts/train_market.py`):
- Batch size: 32
- Window size: 32 time steps
- Learning rate: 5e-5 (with OneCycleLR scheduling)
- Dropout: 0.3
- Early stopping patience: 50 evaluations

The model predicts binary classification (Up/Down) for each asset at each time step.

### Text Classification (Original BDH)

For text classification tasks using the original BDH classifier, refer to the model architecture in `src/bdh/classifier.py`.

## Model Architecture

### Core BDH Model

The BDH model features:
- **Linear Attention** with Rotary Positional Embeddings (RoPE)
- **Conceptual Space Projections** (Expansion/Compression via encoder/decoder)
- **Gated Activation Units** (ReLU-based sparse activations)
- **Modulation Mechanism** (multiplicative gating between attention and MLP paths)

Key components:
- `Attention`: Linear attention mechanism operating in conceptual space
- `BDH`: Full model with embedding, multiple BDH layers, and language model head

### MarketBDH Model

The `MarketBDH` model adapts BDH for financial time series:

1. **Input Projection**: Flattens global market state (`N` assets × `F` features) and projects to embedding dimension `D`
2. **BDH Core**: Processes sequence of market states using linear attention and modulation
3. **Output Projection**: Projects back to asset-specific classification logits (`N` assets × 2 classes)

This approach learns global market dynamics and asset correlations implicitly through latent "particle" interactions.

**Architecture Details**:
- Input: `[Batch, Time, Assets, Features]` → `[B, T, N, F]`
- After projection: `[B, T, D]` where `D` is embedding dimension
- Output: `[B, T, N, 2]` (binary classification per asset per time step)

## Dataset Information

### Market Dataset
- **Training sequences**: ~617 (varies with data)
- **Validation sequences**: ~107 (20% split)
- **Assets**: ~89 (varies with data)
- **Features**: 4 (Positive Momentum, Negative Momentum, Volatility, Normalized Volume)
- **Task**: Binary classification (predicting next-day return direction)

### Data Format

The processed dataset (`market_dataset.pt`) contains:
```python
{
    'X': np.array [Total_Time, N, F],  # Features
    'Y': np.array [Total_Time, N],      # Binary labels (0=down, 1=up)
    'asset_names': list[str]            # Asset ticker symbols
}
```

## Training Improvements

Recent improvements to address overfitting and improve accuracy:

1. **Model Size Reduction**: Reduced from 4 layers/256 dim to 2 layers/128 dim
2. **Regularization**: Increased dropout to 0.3, added weight decay (0.01)
3. **Feature Normalization**: Per-feature normalization using training statistics
4. **Learning Rate Scheduling**: OneCycleLR with warmup
5. **Early Stopping**: Stops training when validation loss stops improving
6. **Gradient Clipping**: Prevents exploding gradients
7. **Label Smoothing**: Reduces overconfidence (0.1 smoothing factor)

## Configuration

Edit `config.yaml` to configure:

```yaml
EODHD:
  api_token: your_token_here

Ingestion:
  parquet_path: "data/parquet"
  default_exchanges: ["NASDAQ", "NYSE"]
  chunk_days: 90
  rate_limit_delay: 1.0  # seconds between requests
  max_retries: 3
  retry_backoff: 2.0
```

## Development

### Adding New Features

- Model architecture: `src/bdh/model.py`
- Market model: `src/bdh/market.py`
- Data processing: `src/dataset/preprocess.py`
- Data ingestion: `src/dataset/ingestion/`

### Debugging

- Use `scripts/debug_data.py` to inspect processed datasets and verify data quality (legacy utility)
- Check parquet files directly using pandas: `pd.read_parquet('data/parquet/nasdaq100/AAPL_US.parquet')`

## License

Copyright 2025 Pathway Technology, Inc.
