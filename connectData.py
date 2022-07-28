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
from pydicom.fileset import FileSet
from os import path as pathOs
from pathlib import Path

filesOrigPath='/workspaces/orig_files_dir.csv'
files_dfPath='/workspaces/konwersjaJsonData/resCSV'

filesOrig= pd.read_csv(filesOrigPath)
files_df = pd.read_csv(files_dfPath)

colName='SOPInstanceUID'
example_study_id='1.3.12.2.1107.5.8.15.100960.30000022021714130657000000014'

filesOrig.loc[filesOrig['StudyInstanceUID'] == example_study_id]
