import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import re, random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses, callbacks, utils, models, Input
from tensorflow.keras import layers as L

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


def data_preprocessing(data_path, num_samples:int =10000):
  input_texts = []
  target_texts = []
  input_characters = set()
  target_characters = set()
  with open(data_path, 'r', encoding='utf-8') as f:
      lines = f.read().split('\n')
  for line in lines[: min(num_samples, len(lines) - 1)]:
      input_text, target_text, _ = line.split('\t')
      input_texts.append(input_text.lower().replace(',',''))
      target_texts.append(target_text)
      for char in input_text:
          if char not in input_characters:
              input_characters.add(char)
      for char in target_text:
          if char not in target_characters:
              target_characters.add(char)

  return input_texts, target_texts


def create_dataset(input_texts, target_texts, num_rows:int, columns:list=["English", "French"]):
    dataset = []
    for e, f in zip(input_texts[:num_rows], target_texts[:num_rows]):
      dataset.append([e.lower(), f.lower()])
    return pd.DataFrame(dataset, columns=["English", "French"])


def preprocess_input(text, tokenizer, max_len):
    seq = [i[0] for i in tokenizer.texts_to_sequences(text)]
    seq = pad_sequences([seq], padding='post', maxlen=max_len)[0]
    return seq