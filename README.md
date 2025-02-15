# Medical Disease Prediction Repository

## Overview
This repository contains code and resources for medical disease prediction using both traditional machine learning and deep learning approaches. It includes training scripts, pre-trained models, and a Flask-based deployment framework.

## Directory Structure
```
├── bert_training
│   ├── BERT_training_notebook.ipynb
│   ├── clinical-stopwords.txt
│   ├── label_encoder.pkl
│   ├── model
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── test.csv
│   ├── tokenizer
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── train.csv
├── flask
│   ├── app.py
│   ├── best_svc_model.joblib
│   ├── clinical-stopwords.txt
│   ├── label_encoder.joblib
│   ├── templates
│   │   └── index.html
│   └── tfidf_vectorizer.joblib
├── testing
│   ├── bert
│   │   ├── label_encoder.pkl
│   │   ├── model
│   │   │   ├── config.json
│   │   │   └── model.safetensors
│   │   └── tokenizer
│   │       ├── special_tokens_map.json
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       └── vocab.txt
│   ├── best_svc_model.joblib
│   ├── clinical-stopwords.txt
│   ├── label_encoder.joblib
│   ├── test images
│   ├── testing_notebook.ipynb
│   └── tfidf_vectorizer.joblib
└── training
    ├── best_svc_model.joblib
    ├── clinical-stopwords.txt
    ├── dataset
    │   ├── disease_data_test.csv
    │   ├── disease_data_train.csv
    │   ├── image_dataset_test
    │   └── image_dataset_train
    ├── label_encoder.joblib
    ├── test.csv
    ├── tfidf_vectorizer.joblib
    ├── train.csv
    └── training_notebook.ipynb
```

## Files and Directories

### `bert_training`
Contains resources and scripts for training and fine-tuning a BERT model for disease prediction.

- **`BERT_training_notebook.ipynb`**: Jupyter notebook demonstrating the training pipeline.
- **Model and Tokenizer**: Pre-trained BERT model files and tokenizer configurations.
- **Dataset**: Training and test CSV files.

### `flask`
Contains the Flask application for model deployment.

- **`app.py`**: Main Flask server script.
- **`templates/index.html`**: Frontend for interacting with the model.
- **Pre-trained Artifacts**: Includes SVM model, label encoder, and TF-IDF vectorizer files.

### `testing`
Resources for testing the trained models on new data.

- **`testing_notebook.ipynb`**: Notebook for testing predictions.
- **BERT model resources**: Includes model and tokenizer files.
- **SVM model resources**: Includes the pre-trained SVM model and vectorizer.

### `training`
Contains resources for training the SVM model.

- **`training_notebook.ipynb`**: Jupyter notebook for the SVM training pipeline.
- **Dataset**: Includes disease data CSVs and image datasets.
- **Pre-trained Artifacts**: Trained SVM model, label encoder, and TF-IDF vectorizer.

## How to Use

### 1. Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Set Up the Environment
Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Train the Models
Use the notebooks in the `training` and `bert_training` directories for training the SVM and BERT models, respectively.

### 4. Test the Models
Run the `testing_notebook.ipynb` in the `testing` directory to evaluate model performance on new data.

### 5. Deploy the Flask Application
Navigate to the `flask` directory and run:
```bash
python app.py
```
Access the application at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Results
- **Machine Learning (SVM)**: Achieved 97% accuracy.
- **Deep Learning (BERT)**: Fine-tuned BERT model for robust text-based classification.

## Acknowledgments
- Datasets sourced from Kaggle.
- Inspired by real-world medical use cases and academic resources.

## Contributors
- **Amith S (22BCE1656)**: Development and Implementation.
- **Duane Ch**: Co-development and Testing.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
