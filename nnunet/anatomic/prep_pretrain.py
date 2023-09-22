""" 
in labels 
1- most central
2 - most peripheral
3 and 4 - middle

we will take unstripped images 
images_structural_unstripped
examples as 3 modalities of choice
    UPENN-GBM-00001_11_FLAIR_unstripped.nii.gz
    UPENN-GBM-00001_11_T1_unstripped.nii.gz
    UPENN-GBM-00001_11_T2_unstripped.nii.gz

from metadata we will take columns
Age_at_scan_years	Survival_from_surgery_days	IDH1

in Survival_from_surgery_days records with values Not Available will be set as -1 
IDH1 - we will encode into ints

"""

import os
from toolz.itertoolz import groupby
import itertools
import SimpleITK as sitk
from os.path import basename, dirname, exists, isdir, join, split
import multiprocessing as mp
from functools import partial
from intensity_normalization.normalize.nyul import NyulNormalize

def get_4_id(masterolds):
    """
    take master id and changes it into string that starts with 0s and have always length 4
    """
    masteroldsStand=str(masterolds)
    if(len(masteroldsStand)==1):
        return f"000{masteroldsStand}"
    elif(len(masteroldsStand)==2):
        return f"00{masteroldsStand}"
    elif(len(masteroldsStand)==3):
        return f"0{masteroldsStand}"
    return masteroldsStand

def prepare_out_paths(id,images_folder,labels_folder):
    #preparing names
    for_id=get_4_id(id)
    label_new_path= f"{labels_folder}/9{for_id}00.nii.gz"
    # prostate_path=join(imagesTrFolder,f"9{for_id}00_000{3}.nii.gz" )
    mod_0=f"{images_folder}/9{for_id}00_000{0}.nii.gz"
    mod_1=f"{images_folder}/9{for_id}00_000{1}.nii.gz"
    mod_2=f"{images_folder}/9{for_id}00_000{2}.nii.gz"
    return (label_new_path,mod_0,mod_1,mod_2)


def set_mask_value(image, mask, value):
    """
    taken from https://discourse.itk.org/t/change-pixel-values-of-images-where-expression/3465
    """
    msk32 = sitk.Cast(mask, sitk.sitkFloat32)
    return sitk.Cast(sitk.Cast(image, sitk.sitkFloat32) *
                     sitk.InvertIntensity(msk32, maximum=1.0) + 
                     msk32*value, image.GetPixelID())

def reg_a_to_b_by_metadata_single_f(fixed_image,moving_image,interpolator):

    # fixed_image=sitk.ReadImage(fixed_image_path)
    # moving_image=sitk.ReadImage(moving_image_path)
    arr=sitk.GetArrayFromImage(moving_image)
    resampled=sitk.Resample(moving_image, fixed_image, sitk.Transform(3, sitk.sitkIdentity), interpolator, 0)
    return resampled    

def normalize_files(image_path,nyul_normalizer):
    image1=sitk.ReadImage(image_path)
    image1 = sitk.DICOMOrient(image1, 'RAS')
    image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data=nyul_normalizer(sitk.GetArrayFromImage(image1))
    #recreating image keeping relevant metadata
    image = sitk.GetImageFromArray(data)  
    image.SetSpacing(image1.GetSpacing())
    image.SetOrigin(image1.GetOrigin())
    image.SetDirection(image1.GetDirection())

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(image_path)
    writer.Execute(image)        

def process_nyul(processed_files,i):
    nyul_normalizer = NyulNormalize()
    #we need to avoid getting too much into normalizer becouse it will lead to memory errors
    files= list(map(lambda el: el[i] ,processed_files))
    images = [sitk.GetArrayFromImage(sitk.ReadImage(image_path)) for image_path in files]  
    nyul_normalizer.fit(images)
    with mp.Pool(processes = mp.cpu_count()) as pool:
        pool.map(partial(normalize_files,nyul_normalizer=nyul_normalizer),files )

def process_upen_gbm_files(paths_pair,images_folder,labels_folder):
    main_dir="/mnt/disks/sdb/raw/upenngbm/upenngbm/UPENN-GBM/UPENN-GBM/NIfTI-files/"
    unstripped= f"{main_dir}/images_structural_unstripped"
    images_segm= f"{main_dir}/images_segm"
    automated_segm= f"{main_dir}automated_segm"

    i,paths_pair=paths_pair
    i=i+500
    loc_im_folder=f"{unstripped}/{paths_pair[0]}"
    loc_seg_file=f"{automated_segm}/UPENN-GBM-{paths_pair[1]}_segm.nii.gz"
    mri_files=os.listdir(loc_im_folder)

    mri_files=list(map(lambda el: f"{loc_im_folder}/{el}",mri_files))
    flair=list(filter(lambda p : "FLAIR_unstripped" in p ,mri_files ))[0]
    t1=list(filter(lambda p : "T1_unstripped" in p ,mri_files ))[0]
    t2=list(filter(lambda p : "T2_unstripped" in p ,mri_files ))[0]
    label_im=sitk.ReadImage(loc_seg_file)
    label_im=set_mask_value(label_im, label_im==3, 2)
    label_im=set_mask_value(label_im, label_im==4, 2)
    label_im=set_mask_value(label_im, label_im==2, 3)
    label_im=sitk.Cast(label_im, sitk.sitkUInt8)
    # saving to files
    label_new_path,mod_0_new_path,mod_1_new_path,mod_2_new_path=prepare_out_paths(i,images_folder,labels_folder)
    
    flair=sitk.Cast(sitk.ReadImage(flair),sitk.sitkFloat32)
    t1=sitk.Cast(sitk.ReadImage(t1),sitk.sitkFloat32)
    t2=sitk.Cast(sitk.ReadImage(t2),sitk.sitkFloat32)
    t1=reg_a_to_b_by_metadata_single_f(flair,t1,sitk.sitkBSpline)
    t2=reg_a_to_b_by_metadata_single_f(flair,t2,sitk.sitkBSpline)
    label_im=reg_a_to_b_by_metadata_single_f(flair,label_im,sitk.sitkNearestNeighbor)
    
    flair=sitk.DICOMOrient(flair, 'RAS')
    t1=sitk.DICOMOrient(t1, 'RAS')
    t2=sitk.DICOMOrient(t2, 'RAS')
    label_im=sitk.DICOMOrient(label_im, 'RAS')



    # imagesTrFolder= join(taskFolder,'imagesTr')
    # labelsTrFolder= join(taskFolder,'labelsTr')
    sitk.WriteImage(label_im, label_new_path)
    sitk.WriteImage(flair, mod_0_new_path)
    sitk.WriteImage(t1,  mod_1_new_path)
    sitk.WriteImage(t2, mod_2_new_path)

    return [label_new_path, mod_0_new_path,mod_1_new_path,mod_2_new_path]

def process_ucsf_files(main_loc_folder,images_folder,labels_folder):
    try:
        main_dir="/mnt/disks/sdb/raw/ucsfdata/ucsf/UCSF-PDGM-v3"
        preprocessed= "/mnt/disks/sdb/anatomic_preprocessed"
        images_folder=f"{preprocessed}/imagesTr"
        labels_folder=f"{preprocessed}/labelsTr"

        i,main_loc_folder=main_loc_folder
        
        i=i+2500
        loc_folder=f"{main_dir}/{main_loc_folder}"
        files=os.listdir( loc_folder )

        files=list(map(lambda el: f"{loc_folder}/{el}",files))
        loc_seg_file=list(filter(lambda p : "tumor_segmentation" in p ,files ))[0]

        flair=list(filter(lambda p : "FLAIR" in p ,files ))[0]
        t1=list(filter(lambda p : "T1.nii.gz" in p ,files ))[0]
        t2=list(filter(lambda p : "T2.nii.gz" in p ,files ))[0]

        label_im=sitk.ReadImage(loc_seg_file)
        label_im=set_mask_value(label_im, label_im==3, 2)
        label_im=set_mask_value(label_im, label_im==4, 2)
        label_im=set_mask_value(label_im, label_im==2, 3)
        label_im=sitk.Cast(label_im, sitk.sitkUInt8)
        # saving to files
        label_new_path,mod_0_new_path,mod_1_new_path,mod_2_new_path=prepare_out_paths(i,images_folder,labels_folder)
        
        flair=sitk.Cast(sitk.ReadImage(flair),sitk.sitkFloat32)
        t1=sitk.Cast(sitk.ReadImage(t1),sitk.sitkFloat32)
        t2=sitk.Cast(sitk.ReadImage(t2),sitk.sitkFloat32)
        t1=reg_a_to_b_by_metadata_single_f(flair,t1,sitk.sitkBSpline)
        t2=reg_a_to_b_by_metadata_single_f(flair,t2,sitk.sitkBSpline)
        label_im=reg_a_to_b_by_metadata_single_f(flair,label_im,sitk.sitkNearestNeighbor)
        
        flair=sitk.DICOMOrient(flair, 'RAS')
        t1=sitk.DICOMOrient(t1, 'RAS')
        t2=sitk.DICOMOrient(t2, 'RAS')
        label_im=sitk.DICOMOrient(label_im, 'RAS')


        # imagesTrFolder= join(taskFolder,'imagesTr')
        # labelsTrFolder= join(taskFolder,'labelsTr')
        sitk.WriteImage(label_im, label_new_path)
        sitk.WriteImage(flair, mod_0_new_path)
        sitk.WriteImage(t1, mod_1_new_path)
        sitk.WriteImage(t2, mod_2_new_path)

        return [label_new_path, mod_0_new_path,mod_1_new_path,mod_2_new_path]
    except :
        return [" "," "," "," "]
def preproc_upen():
    main_dir="/mnt/disks/sdb/raw/upenngbm/upenngbm/UPENN-GBM/UPENN-GBM/NIfTI-files/"
    unstripped= f"{main_dir}/images_structural_unstripped"
    images_segm= f"{main_dir}/images_segm"
    automated_segm= f"{main_dir}automated_segm"
    # rr=[x[0] for x in os.walk(main_dir)]
    rr = [ f.path for f in os.scandir(main_dir) if f.is_dir() ]
    # print(len(os.listdir(unstripped)))
    # print("\n ************** \n")
    # print(len(os.listdir(images_segm)))
    # print(len(os.listdir(automated_segm)))
    def get_label_pair(pathh,list_labels):
        p=pathh.replace('.nii.gz','')
        p=pathh.replace('UPENN-GBM-','')
        l= list(filter(lambda l: p in l ,list_labels))
        # print(f"p {p} l {l}")

        return (pathh,l)
    list_labels=os.listdir(automated_segm)
    list_labels=list(map(lambda ll:  ll.replace('_segm.nii.gz',''),list_labels))
    list_labels=list(map(lambda ll:  ll.replace('UPENN-GBM-',''),list_labels))
    # print(list_labels)
    pairs=list(map( lambda pathh: get_label_pair(pathh,list_labels),os.listdir(unstripped) ))
    pairs= list(filter(lambda pairr: len(pairr[1])>0 , pairs))
    pairs= list(map(lambda pairr: (pairr[0],pairr[1][0]), pairs))

    preprocessed= "/mnt/disks/sdb/anatomic_preprocessed"
    images_folder=f"{preprocessed}/imagesTr"
    labels_folder=f"{preprocessed}/labelsTr"

    # process_upen_gbm_files(pairs[0],images_folder,labels_folder,i=500)
    # print(pairs)
    # print("\n ************** \n")
    # print(os.listdir(unstripped))
    processed_files=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        processed_files=pool.map(partial(process_upen_gbm_files,images_folder=f"{preprocessed}/imagesTr",labels_folder=f"{preprocessed}/labelsTr"), enumerate(pairs))
    
    processed_files= list(filter(lambda p: p!=" ",processed_files))

    process_nyul(processed_files,1)
    process_nyul(processed_files,2)
    process_nyul(processed_files,3)



def preproc_ucsf():
    main_dir="/mnt/disks/sdb/raw/ucsfdata/ucsf/UCSF-PDGM-v3"
    preprocessed= "/mnt/disks/sdb/anatomic_preprocessed"
    images_folder=f"{preprocessed}/imagesTr"
    labels_folder=f"{preprocessed}/labelsTr"
    main_folders=os.listdir(main_dir)


    # process_upen_gbm_files(pairs[0],images_folder,labels_folder,i=500)
    # print(pairs)
    # print("\n ************** \n")
    # print(os.listdir(unstripped))
    with mp.Pool(processes = mp.cpu_count()) as pool:
        processed_files=pool.map(partial(process_ucsf_files,images_folder=f"{preprocessed}/imagesTr",labels_folder=f"{preprocessed}/labelsTr"), enumerate(main_folders))

    processed_files= list(filter(lambda p: p[0]!=" ",processed_files))
    process_nyul(processed_files,1)
    process_nyul(processed_files,2)
    process_nyul(processed_files,3)

# partial(process_upen_gbm_files,images_folder=f"{preprocessed}/imagesTr",labels_folder=f"{preprocessed}/labelsTr")(list(enumerate(pairs))[0])



preproc_upen()
preproc_ucsf()