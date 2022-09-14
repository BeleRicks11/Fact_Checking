'''try:
    ...
except NameError:
    print('UNDEFINED NEEDED VARIABLE(S)')
'''

from os.path import isfile, isdir, join as path_join
import warnings
try:
    from google.colab import drive
except ModuleNotFoundError:
    warnings.warn('COLAB API UNAVAILABLE')
from shutil import unpack_archive, copy
import pickle
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset as torchDataset

def try_get_from_Drive(fname, dpath):
  try:
    if not isdir('/content/drive'): drive.mount('/content/drive')
    if isfile(f'{dpath}{fname}'):
      print('=== Zip found in Google Drive ===')
      copy(f'{dpath}{fname}','.')
  except Exception:
    pass
  
def try_copy_in_Drive(fname, dpath):
  try:
    if not isdir('/content/drive'): drive.mount('/content/drive')
    if not isfile(f'{dpath}{fname}'):
      print('=== copying in Drive ===')
      copy(fname,f'{dpath}{fname}')
      #drive.flush_and_unmount()
  except Exception as e:
    print(e)
    
def unpickle(fname, verbose=False):
    with open(fname, 'rb') as pf:
        if verbose: print(end='loading pickled file... ', flush=True)
        loaded = pickle.load(pf, encoding='bytes')
        if verbose: print('done')
        return loaded


class PrintableDf():
    ''' hack for debugging purposes '''
    def __init__(self, df, colns=[]):
        self.pv = df.copy()
        for n in colns:
            self.pv.iloc[:,n] = self.pv.iloc[:,n].map(lambda _: f'{str(_):20} [...]'.replace('\n',' '))
    def __str__(self):
        s = self.pv.__str__()
        return s
    def __repr__(self):
        return self.__str__()


class autoIntDict(defaultdict):
  ''' Similar to a default dictionary with int values, but keeps count of the newly added
  elements. Accessed elements are automatically added in the dictionary if nonexistent
  with a value that is len(current_dict). For example, given:
    orig = {'ciao':0, 'pippo':1}
    prova = myDict(orig)
    prova['ciao'], prova['hello'], prova['man'], prova.get_newly_added(), prova['man'], prova['ball'], prova
  The last line evaluates to:
    (0, 2, 3, 2, 3, 4, myDict(None, {'ball': 4, 'ciao': 0, 'hello': 2, 'man': 3, 'pippo': 1}))
  '''
  def __init__(self, *aa, **aak):
    self.added_elem_list = []
    self.newly_added = 0
    self.base_len = len(aa[0])
    return super().__init__(None, *aa, **aak)

  def __missing__(self, unused):
    self.newly_added += 1
    self.added_elem_list.append(self.key)
    return self.setdefault(self.key, self.base_len + self.newly_added -1)

  def __getitem__(self, key):
    self.key = key
    return super().__getitem__(key)

  def get_newly_added(self):
    tmp = self.newly_added
    t = self.added_elem_list
    self.base_len += tmp
    self.newly_added = 0
    self.added_elem_list = []
    return t


class CustomTextDataset(torchDataset):
  def __init__(self, df):
    self.evidence = df.iloc[:,0]
    self.claim= df.iloc[:,1]
    self.labels = df.iloc[:,2]
  
  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self, idx):
    return (self.evidence[idx],self.claim[idx], self.labels[idx])
