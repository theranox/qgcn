##########################################################
#                      MNIST RAW DATASET                 #
##########################################################
import os
import numpy as np
import torch
from torch.utils.data import Dataset as RawDataset
import struct
from array import array
import pickle


class MNISTDatasetWrapper(RawDataset):
  """
  Functionality: We use the original dataset as is but after we hit the last
  index, we begin to sample based on inverse probability distribution obtained
  by keeping a count of each class proportion ...
  """
  def __init__(self, dataset, labels):
    self.dataset = dataset
    self.labels = labels

  def __getitem__(self, index):
    data, label = self.dataset[index], self.labels[index]
    return data, label

  def __len__(self):
    return len(self.dataset)


##############################################
#          MNIST Data Loader class           #
##############################################
# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
class MNISTDataLoader(object):
  def __init__(self, training_images_filepath,training_labels_filepath,
                test_images_filepath, test_labels_filepath):
      self.training_images_filepath = training_images_filepath
      self.training_labels_filepath = training_labels_filepath
      self.test_images_filepath = test_images_filepath
      self.test_labels_filepath = test_labels_filepath
  
  def read_images_labels(self, images_filepath, labels_filepath):        
      labels = []
      with open(labels_filepath, 'rb') as file:
          magic, size = struct.unpack(">II", file.read(8))
          if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
          labels = array("B", file.read())        
      
      with open(images_filepath, 'rb') as file:
          magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
          if magic != 2051:
              raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
          image_data = array("B", file.read())        
      images = []
      for i in range(size):
          images.append([0] * rows * cols)
      for i in range(size):
          img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
          img = img.reshape(28, 28)
          images[i][:] = img

      # convert the images and labels ...
      images = np.array(images, dtype=float) # float type ...
      N, H, W = images.shape # extract shape details ...
      images = images.reshape((N, 1, H, W))
      # same for the labels ...
      labels = np.array(labels).reshape( (-1, 1) )
      # convert to torch ...
      images = torch.from_numpy(images).type(torch.FloatTensor)
      labels = torch.from_numpy(labels).type(torch.LongTensor)
      print("images dim:", images.size())
      print("labels dim:", labels.size())
      # return the result ...
      return images, labels

  def load_data(self):
      x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
      x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
      return (x_train, y_train), (x_test, y_test)


######################################################################
#                                                                    #  
#       Process Raw MNIST Data and Convert to Graph                  #
#                                                                    #
######################################################################
def read_cached_graph_dataset( num_train, num_test, dataset_name ):
  # Assertion on dataset splits
  assert num_train != None and isinstance(num_train, int)
  assert num_test != None and isinstance(num_test, int)
  
  # Get data file path
  curr_path = os.path.dirname(os.path.realpath('__file__'))
  base_path = os.path.join(curr_path, "Dataset", "RawGraph")
  if not os.path.exists(base_path):
    print("Dataset with name: {} doesn't exist".format(dataset_name))
    quit()
  
  # get the directory name
  dir_name = "train_{}_test_{}".format(num_train, num_test)
  full_dirpath = os.path.join(base_path, dir_name)
  if not os.path.exists(full_dirpath):
    print("Dataset for num_train={}, num_test={} was not generated".format(num_train, num_test))
    quit()

  # create the dataset name ...
  full_path = os.path.join(full_dirpath, dataset_name + ".pkl")
  assert full_path != None and full_path != "", "Full Path to dataset cannot be None/''"
  print(full_path)

  # confirm that full_path exists ...
  if not os.path.exists(full_path):
    print("Dataset with name: {} doesn't exist".format(dataset_name))
    quit()

  # struct to return ...
  struct = {
      "raw": {
          "x_train_data": [],
          "y_train_data": [],
          "x_test_data" : [],
          "y_test_data" : []
      },
      "geometric": {
          "qgcn_train_data":  [],
          "qgcn_test_data" : [],
          "sgcn_train_data": [],
          "sgcn_test_data" :  [],
      }
  }

  # read data from disk ...
  with open(full_path, "rb") as f:
    struct = pickle.load(f)

  # return the struct ...
  return struct