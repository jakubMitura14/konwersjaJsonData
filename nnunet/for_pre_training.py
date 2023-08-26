# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447

import h5py
from mpi4py import MPI
h5_path='/workspaces/konwersjaJsonData/explore/hdf5_loc/mytestfile.hdf5'
f = h5py.File(h5_path, 'r',driver='mpio', comm=MPI.COMM_WORLD)#,driver='mpio', comm=MPI.COMM_WORLD