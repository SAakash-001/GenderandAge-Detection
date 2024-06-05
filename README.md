# Gender and Age Recognition by Audio (GRA)

## Overview
This project focuses on developing a machine learning system capable of recognizing the gender and age category of individuals based on their voice recordings. The GRA system leverages advanced techniques from the domains of machine learning (ML) and deep learning (DL) to extract discriminative acoustic features from audio signals and employ robust classification models to delineate gender and age-related patterns.

## Motivation
Automatic gender and age recognition has numerous applications, including personalized voice assistants, targeted advertising, customer profiling, and biometric authentication systems. By accurately identifying these demographic attributes, services can be tailored to better meet the needs and preferences of users. Additionally, this project aims to contribute to the ongoing research efforts in the field of audio signal processing and pattern recognition.

## Methodology
The GRA system employs a multi-stage approach to achieve its objectives:

1. **Data Preprocessing**: Voice recordings are preprocessed to remove noise, normalize audio levels, and extract relevant acoustic features.
2. **Feature Engineering**: Various techniques, such as Mel-Frequency Cepstral Coefficients (MFCCs), spectral analysis, and prosodic feature extraction, are employed to derive discriminative acoustic features from the audio signals.
3. **Model Training**: Supervised learning algorithms, including traditional machine learning models (e.g., Support Vector Machines, Random Forests) and deep learning architectures (e.g., Convolutional Neural Networks, Recurrent Neural Networks), are trained on the extracted features to learn patterns associated with gender and age categories.
4. **Model Evaluation**: The trained models are rigorously evaluated using appropriate performance metrics, such as accuracy, precision, recall, and F1-score, to assess their effectiveness and generalization capabilities.
5. **Model Deployment**: The best-performing models are deployed as a web service or integrated into existing applications, enabling real-time gender and age recognition from audio input.

## Pre-Requisites
- `data`:  https://www.kaggle.com/datasets/mozillaorg/common-voice/
- `agengenderrecognitionbyvoice/`: Jupyter Notebooks for data exploration, feature engineering, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.

## Getting Started
1. Clone the repository: `git clone https://github.com/your-username/gra.git`
2. Install the required Python packages: `pip install -r requirements.txt`
3. Explore the Jupyter Notebooks in the `agengenderrecognitionbyvoice/` directory to understand the data, feature engineering techniques, and model training/evaluation processes.
4. Use the scripts in the `agengenderrecognitionbyvoice/` directory to preprocess data, extract features, train models, and perform gender and age recognition on new audio samples.

## Contributing
Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the project's coding standards and guidelines.

## Acknowledgments
- [librosa](https://librosa.org/): A Python library for audio and music analysis.
- [scikit-learn](https://scikit-learn.org/): A machine learning library for Python.
- [PyTorch](https://pytorch.org/): An open-source machine learning framework.
- And all the open-source libraries, datasets, and resources used in this project.
## Results
[code.pdf](https://github.com/user-attachments/files/15580147/code.pdf)
