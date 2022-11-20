import SimpleITK as sitk
import mdai
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import mdai
import math
import time
import mainFuncs
import itertools
import pydicom
import pydicom_seg
import SimpleITK as sitk
import pandas as pd
from os.path import basename, dirname, exists, isdir, join, split
import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.filereader import dcmread
from os import path as pathOs
import itertools
import functools
from itertools import groupby
import pydicom
import pydicom_seg
import SimpleITK as sitk
import pandas as pd
from os.path import basename, dirname, exists, isdir, join, split
import highdicom as hd
import numpy as np
from pydicom.sr.codedict import codes
from pydicom.filereader import dcmread



# resCSVDir='/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
# mainFoldDirMha='/workspaces/konwersjaJsonData/AI4AR_cont'
# mainFoldDirSeg='/workspaces/konwersjaJsonData/AI4AR_dicom'


# mainFoldDir = mainFoldDirMha
# to_dicom_seg= False

# out_files_frame=pd.read_csv(resCSVDir) 

# out_files_frame.columns

# series_desc= np.unique(out_files_frame['series_desc'].to_numpy())


# rowws = list(out_files_frame.iterrows())
# rowws = list(map(lambda el:el[1],rowws))
# groupedByMasterNum = [(key, list(group)) for key,group in groupby(rowws,key = lambda row : row['masterolds'] )]


# masterNum,seriesRows = groupedByMasterNum[0]
# data_path= join(mainFoldDir,"Data",str(masterNum) )
# anat_path= join(mainFoldDir,"Anatomical_Labels",str(masterNum) )
# lesion_path= join(mainFoldDir,"Lesion_Labels",str(masterNum) )
# list(map(lambda pathh: os.makedirs(pathh ,exist_ok = True)  ,[data_path, anat_path,lesion_path]))


# len(groupedByMasterNum)
# rowws


# out_files_frame.columns





client_down_csv='/workspaces/konwersjaJsonData/outCsv/client_files.csv'
client_down_frame=pd.read_csv(client_down_csv) 

client_down_frame.columns




# a = [1,2,3,4,5,6,7]
# aa= groupby(a, key = lambda el : el<4 )
# grouped_anagrams = [(key,list(group)) for key, group in aa]
# grouped_anagrams

# string = 'aaaabbbbbbcccdddddd'
# string_tuple = itertools.groupby(string)
# print(string_tuple, type(string_tuple))

 
# anagrams =['angel','below','glean','bored','robed','study','dusty','cat','act','inch','chin','taste','state','elbow']
# grouped_anagrams = [list(group) for key, group in itertools.groupby(sorted(anagrams, key=sorted),sorted)]
# print(grouped_anagrams)

# for key, iter_item in string_tuple:
#     print(f"{[item for item in  iter_item]}  aa ")
# city_list = [('Decatur', 'AL'), ('Huntsville', 'AL'), ('Selma', 'AL'),
#              ('Anchorage', 'AK'), ('Nome', 'AK'),
#              ('Flagstaff', 'AZ'), ('Phoenix', 'AZ'), ('Tucson', 'AZ'),
#              ]
# def get_state(city_state):
#     return city_state[1]
# cc=list(itertools.groupby(city_list, get_state))
# list(cc[0])
# # for i, j in itertools.groupby( a ,key= lambda el: el%2 ):
# for i, j in cc:
#     print(f" i {i}  {list(j)} ")
# list(bb[1][1])

# selector =[el for sublist in selector for el in sublist]

# list(itertools.compress(a,selector))


# resCSVDir='/workspaces/konwersjaJsonData/outCsv/resCSV.csv'
# out_files_frame=pd.read_csv(resCSVDir) 
# out_files_frame.columns


# L = [('Italy', 'Rome'), ('France', 'Paris'), ('US', 'Washington DC')]
# dd=dict(L)
# items= dd.items()
# (itertools.chain(dd,list))

# list(map(lambda tupl: (f"{tupl[0]}_a", tupl[1])  ,items))