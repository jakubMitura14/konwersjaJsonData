# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830
# https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447

import os
import sys
import requests
import pandas as pd
# from tcia_utils import nbia
import logging
import sys,os
# from .query import TCIAClient, get_response
import pandas as pd
import traceback
import zipfile
import os
import urllib3
import urllib
import sys
import math
import traceback
import sys,os
# from .query import TCIAClient, get_response
import pandas as pd
import traceback
import zipfile
import urllib3, urllib,sys
import concurrent.futures
import ast
import datetime
import numpy as np
# # Check current handlers
# #print(logging.root.handlers)

# # Remove all handlers associated with the root logger object.
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
# #print(logging.root.handlers)

# # Set handler with level = info
# logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
#                     level=logging.INFO)


class TCIAClient:
    GET_IMAGE = "getImage"
    GET_MANUFACTURER_VALUES = "getManufacturerValues"
    GET_MODALITY_VALUES = "getModalityValues"
    GET_COLLECTION_VALUES = "getCollectionValues"
    GET_BODY_PART_VALUES = "getBodyPartValues"
    GET_PATIENT_STUDY = "getPatientStudy"
    GET_SERIES = "getSeries"
    GET_PATIENT = "getPatient"
    GET_SERIES_SIZE = "getSeriesSize"
    CONTENTS_BY_NAME = "ContentsByName"

    def __init__(self, apiKey, baseUrl, resource, maxsize=5):
        self.apiKey = apiKey
        self.baseUrl = baseUrl + "/" + resource
        self.pool_manager = urllib3.PoolManager(maxsize=maxsize)
    def execute(self, url, queryParameters={},preload_content=True):
        queryParameters = dict((k, v) for k, v in queryParameters.items() if v)
        
        headers = {}
        if self.apiKey is not None:
            headers = {"api_key" : self.apiKey }

        queryString = "?%s" % urllib.parse.urlencode(queryParameters)
        requestUrl = url + queryString        
        request = self.pool_manager.request(method='GET', url=requestUrl , headers=headers, preload_content=preload_content)
        return request

    def get_modality_values(self,collection = None , bodyPartExamined = None , modality = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_MODALITY_VALUES
        queryParameters = {"Collection" : collection , "BodyPartExamined" : bodyPartExamined , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_series_size(self, SeriesInstanceUID = None, outputFormat = "json"):
        serviceUrl = self.baseUrl + "/query/" + self.GET_SERIES_SIZE
        queryParameters = {"SeriesInstanceUID" : SeriesInstanceUID, "format" :
                           outputFormat}
        resp = self.execute(serviceUrl, queryParameters)
        return resp

    def contents_by_name(self, name = None):
        serviceUrl = self.baseUrl + "/query/" + self.CONTENTS_BY_NAME
        queryParameters = {"name" : name}
        resp = self.execute(serviceUrl,queryParameters)
        return resp

    def get_manufacturer_values(self,collection = None , bodyPartExamined = None, modality = None , outputFormat = "json"):
        serviceUrl = self.baseUrl + "/query/" + self.GET_MANUFACTURER_VALUES
        queryParameters = {"Collection" : collection , "BodyPartExamined" : bodyPartExamined , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_collection_values(self,outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_COLLECTION_VALUES
        queryParameters = { "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_body_part_values(self,collection = None , bodyPartExamined = None , modality = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_BODY_PART_VALUES
        queryParameters = {"Collection" : collection , "BodyPartExamined" : bodyPartExamined , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_patient_study(self,collection = None , patientId = None , studyInstanceUid = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT_STUDY
        queryParameters = {"Collection" : collection , "PatientID" : patientId , "StudyInstanceUID" : studyInstanceUid , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_series(self, collection = None , modality = None , studyInstanceUID = None, seriesInstanceUID = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_SERIES
        queryParameters = {"Collection" : collection , "StudyInstanceUID": studyInstanceUID, "SeriesInstanceUID" : seriesInstanceUID  , "Modality" : modality , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_patient(self,collection = None , outputFormat = "json" ):
        serviceUrl = self.baseUrl + "/query/" + self.GET_PATIENT
        queryParameters = {"Collection" : collection , "format" : outputFormat }
        resp = self.execute(serviceUrl , queryParameters)
        return resp

    def get_image(self , seriesInstanceUid , downloadPath, zipFileName):
        serviceUrl = self.baseUrl + "/query/" + self.GET_IMAGE
        queryParameters = { "SeriesInstanceUID" : seriesInstanceUid }
        #os.umask(0002)
        try:
            file = os.path.join(downloadPath, zipFileName)
            resp = self.execute( serviceUrl , queryParameters, preload_content=False)
            downloaded = 0
            CHUNK = 256 * 10240
            with open(file, 'wb') as fp:
                for chunk in resp.stream(CHUNK):
                    fp.write(chunk)
        except urllib3.exceptions.HTTPError as e:
            print("HTTP Error:",e.code , serviceUrl)
            return False
        except:
            traceback.print_exc()
            return False

        return True


def get_response(response):
    if response.status == 200:
        return response.data.decode('utf-8')
    else:
        raise ValueError("Error: " + str(response.status))
    
if __name__ == '__main__':
    
    manifest_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]
  
    with open(manifest_file_path,'r') as f:
        content = [x for x in f.read().split('\n') if len(x) > 0]

    i = content.index('ListOfSeriesToDownload=')
    series_instance_uid_list = content[i+1:]

    maxsize = 3
    
    study_data = []
    series_data = []
    tcia_client = TCIAClient(apiKey=None, baseUrl="https://services.cancerimagingarchive.net/services/v3",resource = "TCIA",maxsize=maxsize)
    
    if os.path.exists('data_series.csv'):
        df_series = pd.read_csv('data_series.csv')
    else:
        with concurrent.futures.ThreadPoolExecutor(maxsize) as executor:
            myfutures = {executor.submit(tcia_client.get_series, seriesInstanceUID=x):x for x in series_instance_uid_list}
            for future in concurrent.futures.as_completed(myfutures):
                try:                    
                    response = future.result()
                    r = get_response(response)
                    info_dict = ast.literal_eval(r)[0]
                    series_data.append(info_dict)
                    print(datetime.datetime.now(),'series')
                except:
                    traceback.print_exc()

        df_series = pd.DataFrame(series_data)
        df_series.to_csv('data_series.csv',index=False)

    study_instance_uid_list = np.unique(df_series.StudyInstanceUID.values)

    if os.path.exists('data_study.csv'):
        df_study = pd.read_csv('data_study.csv')
    else:
        with concurrent.futures.ThreadPoolExecutor(maxsize) as executor:
            myfutures = {executor.submit(tcia_client.get_patient_study, studyInstanceUid=x):x for x in study_instance_uid_list}
            for future in concurrent.futures.as_completed(myfutures):
                try:
                    print('study')
                    response = future.result()
                    r = get_response(response)
                    info_dict = ast.literal_eval(r)[0]
                    study_data.append(info_dict)
                    print(datetime.datetime.now(),'study')
                except:
                    traceback.print_exc()

        df_study = pd.DataFrame(study_data)
        df_study.to_csv('data_study.csv',index=False)

    if os.path.exists('data.csv'):
        df = pd.read_csv('data.csv')
    else:
        df = df_series.merge(df_study,
            left_on=['StudyInstanceUID','Collection'], 
            right_on = ['StudyInstanceUID','Collection'], how='left')
        df.to_csv('data.csv',index=False)

    #
    # df.columns
    #
    # SeriesInstanceUID,StudyInstanceUID,Modality,ProtocolName,SeriesDate,SeriesDescription,
    # BodyPartExamined,SeriesNumber,Collection,Manufacturer,ManufacturerModelName,
    # SoftwareVersions,Visibility,ImageCount,PatientID,PatientName,PatientSex,StudyDate,
    # StudyDescription,PatientAge,SeriesCount
    #

    # get unique patient - study - pet/ct pair, get one with large image count, early study date.
    data = {}
    count=0
    for PatientID in np.unique(df.PatientID):

        tmpP = df[df['PatientID']==PatientID]
        
        if PatientID not in data.keys():
            data[PatientID]=[]

        for StudyInstanceUID in np.unique(tmpP.StudyInstanceUID):

            tmpS = df[df['StudyInstanceUID']==StudyInstanceUID]
            
            pet_list = []
            ct_list = []
            # find pet corrected
            tmpPET = tmpS[tmpS['Modality']=='PT']
            for n,row in tmpPET.iterrows():
                if 'nac' in row.SeriesDescription.lower():
                    continue
                if 'uncorrected' in row.SeriesDescription.lower():
                    continue
                if 'cor' in row.SeriesDescription.lower():
                    continue
                if 'sag' in row.SeriesDescription.lower():
                    continue
                if 'mip' in row.SeriesDescription.lower():
                    continue
                if 'PET WB' in row.SeriesDescription:
                    pet_list.append(row)

            if len(pet_list) == 0:
                continue
            if len(pet_list) != 1:
                print(len(pet_list),[x.SeriesDescription for x in pet_list])
                continue

            # find pet corrected
            tmpCT = tmpS[tmpS['Modality']=='CT']
            for n,row in tmpCT.iterrows():
                if 'cor' in row.SeriesDescription.lower():
                    continue
                if 'sag' in row.SeriesDescription.lower():
                    continue
                if 'mip' in row.SeriesDescription.lower():
                    continue
                if 'scout' in row.SeriesDescription.lower():
                    continue
                if 'topogram' in row.SeriesDescription.lower():
                    continue
                ct_list.append(row)

            if len(pet_list) != 1 or len(ct_list) != 1:
                continue
            
    # SeriesInstanceUID,StudyInstanceUID,Modality,ProtocolName,SeriesDate,SeriesDescription,
    # BodyPartExamined,SeriesNumber,Collection,Manufacturer,ManufacturerModelName,
    # SoftwareVersions,Visibility,ImageCount,PatientID,PatientName,PatientSex,StudyDate,
    # StudyDescription,PatientAge,SeriesCount

    # Collection,PatientID,PatientName,PatientSex,StudyInstanceUID,StudyDate,
    # StudyDescription,PatientAge,SeriesCount

            item = dict(
                collection = pet_list[0].Collection,
                patient_id = PatientID,
                patient_sex = pet_list[0].PatientSex,
                patient_age = pet_list[0].PatientAge,
                study_instance_uid = StudyInstanceUID,
                study_date = ct_list[0].StudyDate,
                study_description = pet_list[0].StudyDescription,
                pet_series_description = pet_list[0].SeriesDescription,
                pet_img_count = pet_list[0].ImageCount,
                pet_series_instance_uid = pet_list[0].SeriesInstanceUID,
                ct_series_description = ct_list[0].SeriesDescription,
                ct_img_count = ct_list[0].ImageCount,
                ct_series_instance_uid = ct_list[0].SeriesInstanceUID,
                series_instance_uid = ct_list[0].SeriesInstanceUID,
            )

            data[PatientID].append(item)

    count = 0
    mylist = []
    for k,v in data.items():
        if len(v) == 0:
            continue
        count+=1
        if len(v) > 0:
            v = sorted(v,key=lambda x: x['study_date'])
            mylist.append(v[0])
        else:
            mylist.append(v[0])

    pd.DataFrame(mylist).to_csv(csv_file_path,index=False)
    print(count)

if __name__ == '__main__':
    
    csv_file_path = sys.argv[1]
    root_folder = sys.argv[2]

    if csv_file_path.endswith('.csv'):
        df = pd.read_csv(csv_file_path)
        
    if csv_file_path.endswith('.tcia'):

        with open(csv_file_path,'r') as f:
            content = f.read()
        content = [x for x in content.split('\n') if len(x) > 0]
        i = content.index('ListOfSeriesToDownload=')
        series_instance_uid_list = content[i+1:]
        mylist = []
        for n,uid in enumerate(series_instance_uid_list):
            mylist.append(dict(study_instance_uid='na',series_instance_uid=uid))
        
        df = pd.DataFrame(mylist)
    else:
        raise NotImplementedError()

    for n,row in df.iterrows():
        print(n,len(df))
        study_instance_uid = row.study_instance_uid
        series_instance_uid = row.series_instance_uid
        if study_instance_uid == 'na':
            file_path = os.path.join(root_folder,series_instance_uid,'img.zip')
        else:
            file_path = os.path.join(root_folder,study_instance_uid,series_instance_uid,'img.zip')
        
        if os.path.exists(file_path):
            continue
            
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        folder = os.path.dirname(file_path)
        basename = os.path.basename(file_path)

        tcia_client = TCIAClient(apiKey=None, baseUrl="https://services.cancerimagingarchive.net/services/v3",resource="TCIA")
        tcia_client.get_image(seriesInstanceUid=series_instance_uid,downloadPath=folder,zipFileName=basename)


# python3 /workspaces/konwersjaJsonData/nnunet/pretraining/download_pre.py "/workspaces/konwersjaJsonData/explore/tcia_manifests/CPTAC-PDA_SourceImages_NegativeAssessments-manifest-07-05-2023.tcia" "/workspaces/konwersjaJsonData/explore/tcia_manifests/out_tcia"        