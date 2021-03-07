# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 02:14:21 2021

"""

import gzip
import shutil
import os
import wget
import csv
import linecache
from shutil import copyfile
import ipywidgets as widgets
import numpy as np
import pandas as pd

dataset_URL = "https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/2021-01-20/2021-01-20_clean-dataset.tsv.gz?raw=true" #@param {type:"string"}


#Downloads the dataset (compressed in a GZ format)
#!wget dataset_URL -O clean-dataset.tsv.gz
wget.download(dataset_URL, out='clean-dataset.tsv.gz')

#Unzips the dataset and gets the TSV dataset
with gzip.open('clean-dataset.tsv.gz', 'rb') as f_in:
    with open('clean-dataset.tsv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#Deletes the compressed GZ file
os.unlink("clean-dataset.tsv.gz")

#Gets all possible languages from the dataset
df = pd.read_csv('clean-dataset.tsv',sep="\t")
lang_list = df.lang.unique()
lang_list= sorted(np.append(lang_list,'all'))
lang_picker = widgets.Dropdown(options=lang_list, value="en")

filtered_language = lang_picker.value

#If no language specified, it will get all records from the dataset
if filtered_language == "":
  copyfile('clean-dataset.tsv', 'clean-dataset-filtered.tsv')

#If language specified, it will create another tsv file with the filtered records
else:
  filtered_tw = list()
  current_line = 1
  with open("clean-dataset.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")

    if current_line == 1:
      filtered_tw.append(linecache.getline("clean-dataset.tsv", current_line))

      for line in tsvreader:
        if line[3] == filtered_language:
          filtered_tw.append(linecache.getline("clean-dataset.tsv", current_line))
        current_line += 1

  print('\033[1mShowing first 5 tweets from the filtered dataset\033[0m')
  print(filtered_tw[1:(6 if len(filtered_tw) > 6 else len(filtered_tw))])

  with open('clean-dataset-filtered.tsv', 'w') as f_output:
      for item in filtered_tw:
          f_output.write(item)
          
          