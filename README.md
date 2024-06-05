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
## Acknowledgments
- [NumPy](https://numpy.org/): A fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices.
- [Pandas](https://pandas.pydata.org/): A powerful data analysis and manipulation library for Python, offering data structures and data analysis tools.
- [Matplotlib](https://matplotlib.org/): A comprehensive library for creating static, animated, and interactive visualizations in Python.
- [Seaborn](https://seaborn.pydata.org/): A data visualization library based on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
- [Librosa](https://librosa.org/): A Python library for audio and music analysis, providing building blocks to create music information retrieval systems.
- [TensorFlow](https://www.tensorflow.org/): A comprehensive end-to-end open-source platform for building and deploying machine learning models, including deep learning models.
- [Time](https://docs.python.org/3/library/time.html): A built-in Python module for working with time-related functions and operations.
- [os](https://docs.python.org/3/library/os.html): A built-in Python module providing a way to interact with the operating system, including file and directory operations.
- [pickle](https://docs.python.org/3/library/pickle.html): A built-in Python module for serializing and deserializing Python objects, allowing them to be saved to and loaded from files or other storage mediums.
## Results
<a name="br1"></a> 

import numpy as np # Numerical computation

import pandas as pd # data processing

import matplotlib.pyplot as plt # Data Visualization

import seaborn as sns # Data Visualization

import sklearn # Scientific Computation

import librosa # Music and Audio Files

import tensorflow as tf # Deep Learning

import time # Time Management

import os

import pickle as pkl

from sklearn.model\_selection import train\_test\_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy\_score, precision\_score, recall\_score, f1\_score

from sklearn.linear\_model import LogisticRegression, LinearRegression

from sklearn.svm import SVC

from sklearn.naive\_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, Voting

import warnings

warnings.filterwarnings("ignore")

df = pd.read\_csv(

"../input/common-voice/cv-valid-train.csv"

)

df.head()

**ꢀlename**

**text up\_votes down\_votes age gender accent duration**

cv-valid-train/sample-

000000\.mp3

learn to recognize omens and

follow them the o...

**0**

**1**

**2**

**3**

**4**

1

1

1

1

3

0

0

0

0

2

NaN NaN NaN

NaN NaN NaN

NaN NaN NaN

NaN NaN NaN

NaN NaN NaN

NaN

NaN

NaN

NaN

NaN

cv-valid-train/sample- everything in the universe evolved

000001\.mp3 he said

cv-valid-train/sample- you came so that you could learn

000002\.mp3 about your dr...

cv-valid-train/sample- so now i fear nothing because it

000003\.mp3

was those ome...

cv-valid-train/sample-

000004\.mp3

if you start your emails with

greetings let me...

\# Checking the shape and null records

print("Shape of the data:", df.shape, sep='\n', end='\n\n')

print("Number of Null entries:", df.isna().sum(), sep='\n')

Shape of the data:



<a name="br2"></a> 

(195776, 8)

Number of Null entries:

filename

text

0

0

up\_votes

down\_votes

age

0

0

122008

121717

131065

195776

gender

accent

duration

dtype: int64

\# Different types of age categories

df[df.age.notna()].age.unique().tolist()

['twenties',

'seventies',

'thirties',

'sixties',

'fifties',

'fourties',

'teens',

'eighties']

\# Number of records according to different age categories

sns.set(rc={'figure.figsize':(15, 5)})

sns.countplot(

x = 'age',

data = df[df['age'].notna()],

order = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'sevent

)

plt.show()



<a name="br3"></a> 

\# Different types of gender taken as per the data

df[df.gender.notna()].gender.unique().tolist()

['female', 'male', 'other']

\# Visualizing age with gender

sns.set(rc = {'figure.figsize':(15, 5)})

sns.countplot(

x = 'age',

hue = 'gender',

data = df[df['age'].notna()],

order = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties', 'sevent

)

plt.show()

\# Checking for duplicate values

print("Number of duplicated values:", df.duplicated().sum())

Number of duplicated values: 0



<a name="br4"></a> 

\# Shape of the non null records

print("Data with no null values for the age and gender columns:")

df[df.age.notna() & df.gender.notna()].shape

Data with no null values for the age and gender columns:

(73466, 8)

\# Dropping the unwanted columns and Finalizing the data with non null records

df = df[['filename', 'age', 'gender']]

data = df[df['age'].notna() & df['gender'].notna()]

data.reset\_index(inplace=True, drop=True)

data.head()

**ꢀlename**

**age gender**

**0**

**1**

**2**

**3**

**4**

cv-valid-train/sample-000005.mp3 twenties female

cv-valid-train/sample-000008.mp3 seventies male

cv-valid-train/sample-000013.mp3 thirties female

cv-valid-train/sample-000014.mp3 sixties male

cv-valid-train/sample-000019.mp3

ꢀfties male

\# Cleaned data shape

print("Final Data shaped:")

data.shape

Final Data shaped:

(73466, 3)

\# Converting to numerical data

encoding = {

"gender": {

"female": 0,

"male": 1,

"other": 2,

},

"age": {

"teens": 1,

"twenties": 2,

"thirties": 3,

"fourties": 4,

"fifties": 5,

"sixties": 6,

"seventies": 7,

"eighties": 8,

}



<a name="br5"></a> 

}

data = data.replace(encoding)

data.head()

**ꢀlename age gender**

**0**

**1**

**2**

**3**

**4**

cv-valid-train/sample-000005.mp3

2

7

3

6

5

0

1

0

1

1

cv-valid-train/sample-000008.mp3

cv-valid-train/sample-000013.mp3

cv-valid-train/sample-000014.mp3

cv-valid-train/sample-000019.mp3

We extract the following features:

The following features are related to audio quality through which the model will learn more effectively.

**Spectral Centroid**: each frame of a magnitude spectrogram is normalized and treated as a distribution over

frequency bins, from which the mean (centroid) is extracted per frame

**Spectral Bandwidth**: compute 2nd-order spectral bandwidth

**Spectral Rolloff**: the center frequency for a spectrogram bin such that at least roll\_percent (0.85 by default) of the

energy of the spectrum in this frame is contained in this bin and the bins below

**Mel Frequency Cepstral Coeꢁcients (MFCCs)**: a small set of 20 features that describe the overall shape of a

spectral envelope

Librosa is a Python package for music and audio analysis. It provides the building blocks necessary to create the

music information retrieval systems. Librosa helps to visualize the audio signals and also do the feature

extractions in it using different signal processing techniques.

ds\_path = "/kaggle/input/common-voice/cv-valid-train/"

def feature\_extraction(filename, sampling\_rate=48000):

path = "{}{}".format(ds\_path, filename)

features = list()

audio, \_ = librosa.load(path, sr=sampling\_rate)

gender = data[data['filename'] == filename].gender.values[0]

spectral\_centroid = np.mean(librosa.feature.spectral\_centroid(y=audio, sr=sampling\_

spectral\_bandwidth = np.mean(librosa.feature.spectral\_bandwidth(y=audio, sr=samplin

spectral\_rolloff = np.mean(librosa.feature.spectral\_rolloff(y=audio, sr=sampling\_ra

features.append(gender)

features.append(spectral\_centroid)

features.append(spectral\_bandwidth)



<a name="br6"></a> 

features.append(spectral\_rolloff)

mfcc = librosa.feature.mfcc(y=audio, sr=sampling\_rate)

for el in mfcc:

features.append(np.mean(el))

return features

features = feature\_extraction(data.iloc[0]['filename'])

print("features: ", features)

features: [0, 2147.6058803589067, 2430.4749711924073, 4428.830553016453, -625.28143,

111\.306145, 6.3690877, 34.7671, 31.623457, -4.721562, -0.51193464, -4.9454904,

-12.71285, -2.043672, -3.7277248, -10.708404, -11.206564, -12.003516, -8.506438,

-5.4722967, -4.950396, -3.7100525, -6.3149858, -6.3280854]

def create\_df\_features(orig):

new\_rows = list()

tot\_rows = len(orig)-1

stop\_counter = 55001

for idx, row in orig.iterrows():

if idx >= stop\_counter: break

print("\r", end="")

print("{}/{}".format(idx, tot\_rows), end="", flush=True)

features = feature\_extraction(row['filename'])

features.append(row['age'])

new\_rows.append(features)

return pd.DataFrame(new\_rows, columns=["gender", "spectral\_centroid", "spectral\_ban

"mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6

"mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", "mf

"mfcc17", "mfcc18", "mfcc19", "mfcc20", "label"])

df\_features = create\_df\_features(data[:len(data)//100])

df\_features.head()

733/733

**gender spectral\_centroid spectral\_bandwidth spectral\_rolloff**

**mfcc1**

**mfcc2**

**mfcc3**

**mfcc4**

**0**

**1**

**2**

**3**

**4**

0

1

0

1

1

2147\.605880

2815\.325440

1844\.637736

2123\.711334

2360\.672043

2430\.474971 4428.830553 -625.281433 111.306145 6.369088 34.767101

2451\.922347 4884.633819 -469.936646 126.283371 -16.548635 3.546783

1491\.011525 3164.948048 -418.205475 147.666870 -49.973999 -2.286365

2202\.012929 4111.215965 -464.910706 118.437225 19.749664 27.143229

3

3

2

2957\.220239 4767.080050 -343.833008 157.153870 7.661160 41.898956 -1

5 rows × 25 columns



<a name="br7"></a> 

\# def scale\_features(data):

\#

\#

scaler = StandardScaler()

scaled\_data = scaler.fit\_transform(np.array(data.iloc[:, 0:-1], dtype = float))

\#

return scaled\_data, scaler

\# x, scaler = scale\_features(df\_features)

gender\_mapping = {0: 0, 1: 1, 'other': 2}

df\_features['gender'] = df\_features['gender'].map(gender\_mapping)

\# Gender Model

gender\_X = df\_features.drop(columns=['gender', 'label'])

gender\_y = df\_features['gender']

gender\_X\_train, gender\_X\_test, gender\_y\_train, gender\_y\_test = train\_test\_split(gender\_

gender\_results = pd.DataFrame(columns=['Gender Model', 'Training Time', 'Train Accuracy

gender\_ml\_models = [

('Linear Regression', LinearRegression()),

('Logistic Regression', LogisticRegression()),

('SVM', SVC()),

('Random Forest', RandomForestClassifier()),

('K-NN', KNeighborsClassifier()),

('Gaussian Naive Bayes', GaussianNB()),

('Bernoulli Naive Bayes', BernoulliNB()),

('Decision Tree', DecisionTreeClassifier())

]

for name, model in gender\_ml\_models:

start\_time = time.time()

model.fit(gender\_X\_train, gender\_y\_train)

with open(f'models/gender\_{name}.pkl', 'wb') as file:

pkl.dump(model, file)

end\_time = time.time()

train\_time = end\_time - start\_time

gender\_y\_pred\_train = np.round(model.predict(gender\_X\_train))

gender\_y\_pred\_test = np.round(model.predict(gender\_X\_test))

train\_acc = accuracy\_score(gender\_y\_train, gender\_y\_pred\_train)

test\_acc = accuracy\_score(gender\_y\_test, gender\_y\_pred\_test)

gender\_results.loc[len(gender\_results)] = {

'Gender Model': name,

'Training Time': train\_time,

'Train Accuracy': train\_acc,

'Test Accuracy': test\_acc}



<a name="br8"></a> 

gender\_dl\_models = [

(

'Simple Dense 1',

tf.keras.models.Sequential([

tf.keras.layers.Dense(32, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

(

'Simple Dense 2',

tf.keras.models.Sequential([

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

(

'Simple Dense 3',

tf.keras.models.Sequential([

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

]

for name, model in gender\_dl\_models:

model.compile(optimizer='adam', loss='binary\_crossentropy', metrics=['accuracy'])

start\_time = time.time()

model.fit(gender\_X\_train, gender\_y\_train, verbose=0, epochs=10, batch\_size=32)

model.save(f"models/gender\_{name}.h5")

end\_time = time.time()

train\_time = end\_time - start\_time

train\_loss, train\_acc = model.evaluate(gender\_X\_train, gender\_y\_train, verbose=0)

test\_loss, test\_acc = model.evaluate(gender\_X\_test, gender\_y\_test, verbose=0)

gender\_y\_pred\_test = (model.predict(gender\_X\_test, verbose=0) > 0.5).astype('int32'

gender\_results.loc[len(gender\_results)] = {

'Gender Model': name,

'Training Time': train\_time,

'Train Accuracy': train\_acc,

'Test Accuracy': test\_acc}

gender\_results



<a name="br9"></a> 

**Gender Model Training Time Train Accuracy Test Accuracy**

**0**

**1**

Linear Regression

Logistic Regression

SVM

0\.005053

0\.124435

0\.017072

0\.369453

0\.002243

0\.003474

0\.003609

0\.016823

1\.583714

1\.951585

2\.250155

0\.863714

0\.824532

0\.770017

1\.000000

0\.790460

0\.829642

0\.780239

1\.000000

0\.741056

0\.771721

0\.780239

0\.836735

0\.836735

0\.768707

0\.850340

0\.721088

0\.843537

0\.816327

0\.829932

0\.741497

0\.775510

0\.782313

**2**

**3**

Random Forest

K-NN

**4**

**5**

Gaussian Naive Bayes

Bernoulli Naive Bayes

Decision Tree

**6**

**7**

**8**

Simple Dense 1

Simple Dense 2

Simple Dense 3

**9**

**10**

\# Age Model

age\_X = df\_features.drop(columns=['gender', 'label'])

age\_y = df\_features['label']

age\_X\_train, age\_X\_test, age\_y\_train, age\_y\_test = train\_test\_split(age\_X, age\_y, test\_

age\_results = pd.DataFrame(columns=['age Model', 'Training Time', 'Train Accuracy', 'Te

age\_ml\_models = [

('Linear Regression', LinearRegression()),

('Logistic Regression', LogisticRegression()),

('SVM', SVC()),

('Random Forest', RandomForestClassifier()),

('K-NN', KNeighborsClassifier()),

('Gaussian Naive Bayes', GaussianNB()),

('Bernoulli Naive Bayes', BernoulliNB()),

('Decision Tree', DecisionTreeClassifier())

]

for name, model in age\_ml\_models:

start\_time = time.time()

model.fit(age\_X\_train, age\_y\_train)

with open(f'models/age\_{name}.pkl', 'wb') as file:

pkl.dump(model, file)

end\_time = time.time()

train\_time = end\_time - start\_time

age\_y\_pred\_train = np.round(model.predict(age\_X\_train))

age\_y\_pred\_test = np.round(model.predict(age\_X\_test))

train\_acc = accuracy\_score(age\_y\_train, age\_y\_pred\_train)

test\_acc = accuracy\_score(age\_y\_test, age\_y\_pred\_test)

age\_results.loc[len(age\_results)] = {

'age Model': name,



<a name="br10"></a> 

'Training Time': train\_time,

'Train Accuracy': train\_acc,

'Test Accuracy': test\_acc}

age\_dl\_models = [

(

'Simple Dense 1',

tf.keras.models.Sequential([

tf.keras.layers.Dense(32, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

(

'Simple Dense 2',

tf.keras.models.Sequential([

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

(

'Simple Dense 3',

tf.keras.models.Sequential([

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(64, activation='relu'),

tf.keras.layers.Dense(1, activation='sigmoid')

])

),

]

for name, model in age\_dl\_models:

model.compile(optimizer='adam', loss='binary\_crossentropy', metrics=['accuracy'])

start\_time = time.time()

model.fit(age\_X\_train, age\_y\_train, verbose=0, epochs=10, batch\_size=32)

end\_time = time.time()

model.save(f"models/age\_{name}.h5")

train\_time = end\_time - start\_time

train\_loss, train\_acc = model.evaluate(age\_X\_train, age\_y\_train, verbose=0)

test\_loss, test\_acc = model.evaluate(age\_X\_test, age\_y\_test, verbose=0)

age\_y\_pred\_test = (model.predict(age\_X\_test, verbose=0) > 0.5).astype('int32')

age\_results.loc[len(age\_results)] = {

'age Model': name,

'Training Time': train\_time,

'Train Accuracy': train\_acc,

'Test Accuracy': test\_acc}



<a name="br11"></a> 

age\_results

**age Model Training Time Train Accuracy Test Accuracy**

**0**

**1**

Linear Regression

Logistic Regression

SVM

0\.006228

0\.090292

0\.047616

0\.400267

0\.002302

0\.003671

0\.003412

0\.014764

1\.561944

1\.872699

2\.225027

0\.223169

0\.342419

0\.345826

1\.000000

0\.488927

0\.420784

0\.390119

1\.000000

0\.066440

0\.066440

0\.066440

0\.251701

0\.319728

0\.312925

0\.360544

0\.265306

0\.333333

0\.333333

0\.319728

0\.074830

0\.074830

0\.074830

**2**

**3**

Random Forest

K-NN

**4**

**5**

Gaussian Naive Bayes

Bernoulli Naive Bayes

Decision Tree

**6**

**7**

**8**

Simple Dense 1

Simple Dense 2

Simple Dense 3

**9**

**10**

os.mkdir('models')

import shutil

def zip\_folder(folder\_path, zip\_path):

shutil.make\_archive(zip\_path, 'zip', folder\_path)

folder\_path = 'models'

zip\_path = 'zipped'

zip\_folder(folder\_path, zip\_path)


