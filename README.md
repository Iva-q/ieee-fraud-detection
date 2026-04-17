# IEEE-CIS Fraud Detection: End-to-End ML Pipeline

> 🚧 **Work in progress.** Full write-up, results, and demo will appear here as the project develops.

End-to-end solution for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) Kaggle competition:
from EDA to a production-ready inference service.

## Stack

- **ML**: LightGBM, CatBoost, scikit-learn
- **Validation**: time-based K-Fold (no leakage)
- **Feature engineering**: aggregations, frequency & target encoding, D-column tricks
- **Interpretability**: SHAP
- **Experiment tracking**: MLflow
- **Hyperparameter tuning**: Optuna (TPE sampler)
- **Serving**: FastAPI + Docker
- **CI**: GitHub Actions (ruff + pytest)

## Project status

- [x] Repository setup
- [ ] Data loading & EDA
- [ ] Time-based CV strategy
- [ ] Baseline model
- [ ] Feature engineering
- [ ] Hyperparameter tuning
- [ ] Ensemble
- [ ] SHAP analysis
- [ ] FastAPI service
- [ ] Docker
- [ ] Final README with results

## Setup

```bash
conda create -n ieee-fraud python=3.11 -y
conda activate ieee-fraud
pip install -e ".[dev,serve]"
```

## License

MIT