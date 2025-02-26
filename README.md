# Audio_Classification_with_Hugging_Face_Transformer
# Introduction
Identification of speech commands, also known as keyword spotting (KWS), is important from an engineering perspective for a wide range of applications, from indexing audio databases and indexing keywords, to running speech models locally on microcontrollers. Currently, many human-computer interfaces (HCI) like Google Assistant, Microsoft Cortana, Amazon Alexa, Apple Siri and others rely on keyword spotting. There is a significant amount of research in the field by all major companies, notably Google and Baidu.

In the past decade, deep learning has led to significant performance gains on this task. Though low-level audio features extracted from raw audio like MFCC or mel-filterbanks have been used for decades, the design of these low-level features are flawed by biases. Moreover, deep learning models trained on these low-level features can easily overfit to noise or signals irrelevant to the task. This makes it is essential for any system to learn speech representations that make high-level information, such as acoustic and linguistic content, including phonemes, words, semantic meanings, tone, speaker characteristics from speech signals available to solve the downstream task. Wav2Vec 2.0, which solves a self-supervised contrastive learning task to learn high-level speech representations, provides a great alternative to traditional low-level features for training deep learning models for KWS.

In this notebook, we train the Wav2Vec 2.0 (base) model, built on the Hugging Face Transformers library, in an end-to-end fashion on the keyword spotting task and achieve state-of-the-art results on the Google Speech Commands Dataset.
# Requirements
```
Python 3.7+
Jupyter Notebook
Hugging Face Transformers
TensorFlow or PyTorch
Additional libraries as specified in requirements.txt
```
# Installation
> Clone the repository:
```
git clone https://github.com/sumya24/Audio_Classification_with_Hugging_Face_Transformer
.git
cd sumya24
```
> Install the required packages:
```
pip install -r requirements.txt
```
# Usage
> Open the Jupyter Notebook:
```
jupyter notebook g55.ipynb
```
> Run the cells sequentially to preprocess data, train the model, and evaluate its performance.

# Notebook Overview
### Data Loading: Loading and preprocessing the audio dataset.
### Model Training: Training the Wav2Vec2 model using Keras.
### Evaluation: Evaluating the model's performance on the test set.

# Results
This project has successfully demonstrated the potential of leveraging advanced machine learning models, specifically Wav2Vec2, for audio classification tasks within the realm of educational technology and data science. By implementing this cutting-edge technology. The project achieved high accuracy in identifying spoken commands, demonstrating robust performance across various metrics like precision, recall, and F1-score. These results un-derscore the model’s effectiveness in enhancing audio analysis beyond traditional meth-ods. Future research may refine the model further and explore practical applications such as integration into voice-controlled devices, thereby expanding its utility across diverse industries reliant on accurate speech recognition solutions.
![Edit button](images/result.png)

# Contributing
### Saiprasad Patil
<ul>
<li>Github: https://github.com/SaiprasadPatil</li>
  <li>Email: <a href="mailto:Saiprasad6767@gmail.com">Saiprasad6767@gmail.com</a></li>
</ul>

### Ajay Magaonkar
<ul>
<li>Github: https://github.com/adm2425</li>
  <li>Email: <a href="mailto:ajaymangaonkar8@gmail.com">28mahakjain2000@gmail.com</a></li>
</ul>
# Dataset
In this Project we used this dataset [Iemocap-full-release Dataset](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease/).

# Keras Code Link
[Keras Code ](https://keras.io/examples/audio/wav2vec2_audiocls/).
