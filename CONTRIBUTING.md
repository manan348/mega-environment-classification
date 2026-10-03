# Contributing to Mega-Environment Classification

Thank you for your interest in contributing! This project classifies Punjab districts into mega-environments for wheat variety recommendations.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Set up Google Earth Engine authentication (see below)
5. Create `config/local_settings.yaml` with your GEE project ID

## GEE Setup

1. Sign up at https://earthengine.google.com/
2. Create a Google Cloud Project
3. Enable Earth Engine API
4. Run `earthengine authenticate`
5. Add project ID to `config/local_settings.yaml`

## Running the Pipeline

```bash
# Full pipeline
python -m src.mega_env.pipeline --phase all

# Individual phases
python -m src.mega_env.pipeline --phase 1  # Extraction
python -m src.mega_env.pipeline --phase 2  # PCA
python -m src.mega_env.pipeline --phase 3  # Clustering
python -m src.mega_env.pipeline --phase 4  # Validation
python -m src.mega_env.pipeline --phase 5  # Recommendation templates
```

## Development

- Follow PEP 8 style guide
- Add type hints to new functions
- Write tests for new functionality
- Update documentation as needed

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Run tests (if available)
4. Update README if needed
5. Submit PR with clear description

## Code Style

- Use 4 spaces for indentation
- Max line length: 100 characters
- Use descriptive variable names
- Add docstrings to public functions

## Issues

Please use GitHub Issues for:
- Bug reports
- Feature requests
- Questions about usage

## License

By contributing, you agree that your contributions will be licensed under the MIT License.