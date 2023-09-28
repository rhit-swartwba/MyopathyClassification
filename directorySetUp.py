
#SCRIPT FOR DIRECTORIES
# import directories

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories
#change ending of dataset home for different test

dataset_home = '/Users/blaiseswartwood/Downloads/Dataprep/Spectrogram/'
subdirs = ['train/', 'test/', 'final/']

for subdir in subdirs:
  # create label subdirectories
  labeldirs = ['normal/', 'myo/']
  for labldir in labeldirs:
     newdir = dataset_home + subdir + labldir
     makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pics for validation
val_ratio = 0.2
# define ratio of pics for test
final_ratio = 0.1
# copy dataset images into subdirectories
src_directory = '/Users/blaiseswartwood/Downloads/Created Images/Spectrogram'
for file in listdir(src_directory):
  src = src_directory + '/' + file
  dst_dir = 'train/'
  if random() > final_ratio and random() < val_ratio:
     dst_dir = 'test/'
  elif random() <= final_ratio:
     dst_dir = 'final/'
  if file.startswith('N'):
     dst = dataset_home + dst_dir + 'normal/'  + file
     copyfile(src, dst)
  elif file.startswith('M'):
     dst = dataset_home + dst_dir + 'myo/'  + file
     copyfile(src, dst)
