#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# copied defoult preprocessor
import comet_ml
import multiprocessing
import optuna
import shutil
from time import sleep
from typing import Union, Tuple

import nnunetv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder, get_filenames_of_train_images_and_targets
from tqdm import tqdm
from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import (PreprocessingSettings, Sample,
                                      resample_to_reference_scan)
from nnunetv2.utilities.label_handling.label_handling import LabelManager
import monai
import SimpleITK as sitk
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass
import lapgm
import tempfile

from pathlib import Path
from .anatomy_model.nnUNetTrainer.Main_trainer_pl import *
import torchio as tio
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks import eval_mode
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    DivisiblePadd,
    KeepLargestConnectedComponent,
    Lambdad,
    LoadImaged,
    MapTransform,
    RandAffined,
    ScaleIntensityd,
    Rand3DElasticd,
    AdjustContrastd,
    BatchInverseTransform,
)
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import first, set_determinism
from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    pad_list_data_collate,
    TestTimeAugmentation
)

import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch, pad_list_data_collate
from monai.transforms.compose import Compose
from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.post.dictionary import Invertd
from monai.transforms.transform import Randomizable
from monai.transforms.utils_pytorch_numpy_unification import mode, stack
from monai.utils import CommonKeys, PostFix, optional_import
set_determinism(seed=0)



if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

__all__ = ["TestTimeAugmentation"]

DEFAULT_POST_FIX = PostFix.meta()


def _identity(x):
    return x


lapgm.use_gpu(True)# on cpu 16 9 iters
debias_obj = lapgm.LapGM(downscale_factor=1)
debias_obj.set_hyperparameters(tau=0.0426040566281932, n_classes=12, log_initialize=False)


class AnatomyPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        data, seg = self.run_case_npy(data, seg, data_properites, plans_manager, configuration_manager,
                                      dataset_json)
        return data, seg, data_properites

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError('Unable to locate class \'%s\' for normalization' % scheme)
            normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg






def orientt(path):
    image=sitk.ReadImage(path)
    image=sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    image=sitk.Cast(image, sitk.sitkFloat32)
    return image

def save_into_temp(tupl,temp_dir):
    image,orig_dir=tupl
    writer = sitk.ImageFileWriter()
    stemm=Path(orig_dir).stem.replace('.nii','')
    newPath=f"{temp_dir}/{stemm}.nii.gz"
    writer.SetFileName(newPath)
    writer.Execute(image)
    return newPath




def get_im_from_array(arr,channel,orig_im):
    arr=arr[channel,:,:,:]
    image = sitk.GetImageFromArray(arr)  
    image.SetSpacing(orig_im.GetSpacing())
    image.SetOrigin(orig_im.GetOrigin())
    image.SetDirection(orig_im.GetDirection())    
    return image

def return_corrected(norm_str,arrrr,t2w_image,adc_image,hbv_image,arrrr_debiased):
    
    if(norm_str=="t2w_adc_hbv"):
        return [get_im_from_array(arrrr,0,t2w_image), get_im_from_array(arrrr,1,adc_image), get_im_from_array(arrrr,2,hbv_image)]
    if(norm_str=="t2w_adc"):
        return [get_im_from_array(arrrr,0,t2w_image), get_im_from_array(arrrr,1,adc_image),get_im_from_array(arrrr_debiased,2,hbv_image)]
    if(norm_str=="t2w_hbv"):
        return [get_im_from_array(arrrr,0,t2w_image), get_im_from_array(arrrr_debiased,1,adc_image), get_im_from_array(arrrr,2,hbv_image)]
    if(norm_str=="t2w"):
        return [get_im_from_array(arrrr,0,t2w_image),get_im_from_array(arrrr_debiased,1,adc_image), get_im_from_array(arrrr_debiased,2,hbv_image)]



def get_paramss(arrrr):
    try:
        return debias_obj.estimate_parameters(arrrr, print_tols=True)
    except:
        return debias_obj.estimate_parameters(arrrr, print_tols=True)
        
            
def remove_small(arr):
    sm=0.0001
    ind=(arr<sm)
    arr[ind]=sm
    return arr     
       
### bias field correction and normalization
#on the basis of https://github.com/lucianoAvinas/lapgm/blob/main/examples/image_correction.ipynb
def bias_field_and_normalize_help(t2w_image,adc_image,hbv_image):
    # Approximate location of farthest peak for true data.
    # In practice this can be set to any fixed value of choice.
    TRGT = 1.0
    #first bias field correction
    modalities_to_normalize=  [t2w_image,adc_image,hbv_image]
    modalities_to_normalize = list(map(sitk.GetArrayFromImage ,modalities_to_normalize))
    modalities_to_normalize = list(map(lambda arr : np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(float) ,modalities_to_normalize))
    print(f"minnnnnn in bias corr {np.min(np.stack(modalities_to_normalize))}")
    modalities_to_normalize = list(map(remove_small,modalities_to_normalize))

    # print(f"aaaaaaaaaaaaaaaaaaaaa {np.sum((np.stack(modalities_to_normalize)==0).flatten())} {np.min(np.stack(modalities_to_normalize).flatten())} ")
    #80830
    arrrr = lapgm.to_sequence_array(modalities_to_normalize)
    # arrrr=np.nan_to_num(arrrr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    nan_summ=np.sum(np.isnan(arrrr))
    if(nan_summ>0):
        print(f"eeeeeeeeeeeeeerrorrr naaans")
        raise Exception('nanns')   
    
    params = get_paramss(arrrr)
    # Run debias procedure and take parameter output
    arrrr= lapgm.debias(arrrr, params)
    arrrr_debiased=arrrr.copy()

    to_norm="t2w_adc_hbv"  
    arrrr=np.nan_to_num(arrrr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)    
    #we need new parameters only if we are normalizing less than bias field correcting

    arrrr = lapgm.normalize(arrrr, params, target_intensity=TRGT)
    arrrr=np.nan_to_num(arrrr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    return return_corrected(to_norm,arrrr,t2w_image,adc_image,hbv_image,arrrr_debiased)


def z_score_normalize(image_orig):
    arr=sitk.GetArrayFromImage(image_orig)
    mean = arr.mean()
    std = arr.std()
    arr = (arr - mean) / (max(std, 1e-8))
    
    image = sitk.GetImageFromArray(arr)      
    image.SetSpacing(image_orig.GetSpacing())
    image.SetOrigin(image_orig.GetOrigin())
    image.SetDirection(image_orig.GetDirection()) 
    return image
    
def get_modalities_to_norm_classic(norm_str, t2w_image,adc_image,hbv_image):
    
    if(norm_str=="t2w_adc_hbv"):
        return [z_score_normalize(t2w_image), z_score_normalize(adc_image), z_score_normalize(hbv_image)]
    if(norm_str=="t2w_adc"):
        return [z_score_normalize(t2w_image), z_score_normalize(adc_image), hbv_image]
    if(norm_str=="t2w_hbv"):
        return [z_score_normalize(t2w_image), adc_image, z_score_normalize(hbv_image)]
    if(norm_str=="t2w"):
        return [z_score_normalize(t2w_image), adc_image, hbv_image]

def bias_field_and_normalize_help_sitk(t2w_image,adc_image,hbv_image):
    print("bias field correction by sitk")
    norm_str="t2w_adc_hbv"
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    t2w_image= corrector.Execute(t2w_image)
    adc_image= corrector.Execute(adc_image)
    hbv_image= corrector.Execute(hbv_image)
    return get_modalities_to_norm_classic(norm_str, t2w_image,adc_image,hbv_image)

def bias_field_and_normalize_help_b(t2w_image,adc_image,hbv_image):
    try:
        return bias_field_and_normalize_help(t2w_image,adc_image,hbv_image)
    except:
        return bias_field_and_normalize_help(t2w_image,adc_image,hbv_image) 


def bias_field_and_normalize(t2w_image,adc_image,hbv_image):
    try:
        return bias_field_and_normalize_help_b(t2w_image,adc_image,hbv_image)
    except:
        return bias_field_and_normalize_help_sitk(t2w_image,adc_image,hbv_image) 

def get_pred_one_hot(output,is_regions):
    if(is_regions):
        predicted_segmentation_onehot = (output > 0.5).long()
        return predicted_segmentation_onehot
    else:
        
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        return predicted_segmentation_onehot







def save_label(output,chann,name,path_of_example,tempdir):
    label_image=get_im_from_array(output,chann,sitk.ReadImage(path_of_example))
    writer = sitk.ImageFileWriter()   
    newPath=f"{tempdir}/{name}.nii.gz"
    writer.SetFileName(newPath)
    writer.Execute(label_image)
 
def min_0(el):
    if(el<0):
        return 0
    return el

def max_second(size_zyx,curr_size,i):
    if(size_zyx[i]>(curr_size[i]-1)):
        return curr_size[i]
    return size_zyx[i]


def my_center_crop(image,physical_size):
    spacing_zyx = list(image.GetSpacing())[::-1]
    size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
    size_zyx = [int(np.round(x)) for x in size_zyx] 
    curr_size = list(image.GetSize())[::-1]

    size_zyx= list(map(lambda i:max_second(size_zyx,curr_size,i) ,list(range(3))))

    beg=[int((curr_size[0]-size_zyx[0])//2),int((curr_size[1]-size_zyx[1])//2),int((curr_size[2]-size_zyx[2])//2)]
    beg = list(map(min_0,beg))
    # print(f"aaaaaaa curr_size {curr_size}  size_zyx {size_zyx} beg {beg}  physical_size {physical_size}")

    extract = sitk.ExtractImageFilter()
    extract.SetSize([size_zyx[2],size_zyx[1],size_zyx[0]])
    extract.SetIndex([beg[2],beg[1],beg[0]])
    extracted_image = extract.Execute(image)
    return extracted_image



def un_crop(segmentation_reverted_cropping,arr,channel,slicer,plans_manager):
    segmentation_reverted_cropping[slicer] = arr[channel,:,:,:]    
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    return segmentation_reverted_cropping

def my_convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = 1):
    # predicted_logits=predicted_logits.cpu().numpy()
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    
    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros([predicted_logits.shape[0]]+list(properties_dict['shape_before_cropping']),
                                              dtype=float)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])

    # print(f"asdasdasdddddddddddd {slicer}")

    segmentation_reverted_cropping[:,slicer[0],slicer[1],slicer[2]] = predicted_logits   
    # print(f"11 rrrrrrrrrr 0: {np.sum(predicted_logits[0,:,:,:])}  1: {np.sum(predicted_logits[1,:,:,:])}  2: {np.sum(predicted_logits[2,:,:,:])}  3: {np.sum(predicted_logits[3,:,:,:])} ")
    # print(f"to tttt {plans_manager.transpose_backward }")
    tr=plans_manager.transpose_backward
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(tuple([0,tr[0]+1,tr[1]+1,tr[2]+1] ))
    return segmentation_reverted_cropping
    # res= list(map(lambda chan :un_crop(segmentation_reverted_cropping,predicted_logits,chan,slicer,plans_manager), list(range(predicted_logits.shape[0] ))))
    # res= np.stack(res)
    # return res

    # predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        




def get_largest_connected_component(binary_image):
    if(np.sum(binary_image.flatten())==0):
        return binary_image
    binary_image=sitk.GetImageFromArray(binary_image.astype(np.uint8))
    #taken from https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958
    # 1. Convert binary image into a connected component image, each component has an integer label.
    # 2. Relabel components so that they are sorted according to size (there is an
    #    optional minimumObjectSize parameter to get rid of small components).
    # 3. Get largest connected componet, label==1 in sorted component image.
    component_image = sitk.ConnectedComponent(binary_image)
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    return sitk.GetArrayFromImage(largest_component_binary_image)


def execute_infrence(data,clinical,network,is_swin_monai):
    if(is_swin_monai):
        # monai.inferers.SlidingWindowInfererAdapt((64, 192, 160))(torch.tensor(data).float().cuda(), network,clinical=clinical  )[0]  
        return monai.inferers.sliding_window_inference( (torch.tensor(data).float().cuda()), (64, 192, 160), 1, network,clinical=clinical)[0]  
    return monai.inferers.sliding_window_inference(torch.tensor(data).float().cuda(), (48, 192, 160), 1, network)[0]  
    # return monai.inferers.SlidingWindowInfererAdapt((48, 192, 160))(torch.tensor(data).float().cuda(), network  )[0]  

def groupByMaster(rowws):
    grouped_by_master= groupby(lambda row : row[1]['masterolds'],rowws)
    # grouped_by_master=[(key,list(group)) for key, group in grouped_by_master]
    return dict(grouped_by_master).items()

def my_to_float(strr):
    """
    if possible cast string to float
    """
    if(type(strr) is float):
        return strr
    strr= strr.replace(",",".")
    return float(strr)
    

class My_TestTimeAugmentation:
    """
    adapted to ensemble from https://github.com/Project-MONAI/MONAI/blob/77b175986a063b82f0147eb311579003b4ed3569/monai/data/test_time_augmentation.py#L50
    """

    def __init__(
        self,
        transform: InvertibleTransform,
        batch_size: int,
        plans
        ,configuration
        ,fold,dataset_json
        ,is_swin_monai
        ,is_classic_nnunet
        ,checkpoint_paths,
        num_workers: int = 0,
        # inferrer_fn: Callable = _identity,
        device: str | torch.device = "cpu",
        image_key=CommonKeys.IMAGE,
        orig_key=CommonKeys.LABEL,
        nearest_interp: bool = True,
        orig_meta_keys: str | None = None,
        meta_key_postfix=DEFAULT_POST_FIX,
        to_tensor: bool = True,
        output_device: str | torch.device = "cpu",
        post_func: Callable = _identity,
        return_full_data: bool = False,
        progress: bool = True,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.inferrer_fn = inferrer_fn
        self.device = device
        self.image_key = image_key
        self.return_full_data = return_full_data
        self.progress = progress
        self._pred_key = CommonKeys.PRED
        self.inverter = Invertd(
            keys=self._pred_key,
            transform=transform,
            orig_keys=orig_key,
            orig_meta_keys=orig_meta_keys,
            meta_key_postfix=meta_key_postfix,
            nearest_interp=nearest_interp,
            to_tensor=to_tensor,
            device=output_device,
            post_func=post_func,
        )

        # check that the transform has at least one random component, and that all random transforms are invertible
        self._check_transforms()
        self.plans=plans
        self.configuration=configuration
        self.fold=fold
        self.dataset_json=dataset_json
        self.is_swin_monai=is_swin_monai
        self.is_classic_nnunet=is_classic_nnunet
        self.checkpoint_paths=checkpoint_paths


    def _check_transforms(self):
        """Should be at least 1 random transform, and all random transforms should be invertible."""
        ts = [self.transform] if not isinstance(self.transform, Compose) else self.transform.transforms
        randoms = np.array([isinstance(t, Randomizable) for t in ts])
        invertibles = np.array([isinstance(t, InvertibleTransform) for t in ts])
        # check at least 1 random
        if sum(randoms) == 0:
            warnings.warn(
                "TTA usually has at least a `Randomizable` transform or `Compose` contains `Randomizable` transforms."
            )
        # check that whenever randoms is True, invertibles is also true
        for r, i in zip(randoms, invertibles):
            if r and not i:
                warnings.warn(
                    f"Not all applied random transform(s) are invertible. Problematic transform: {type(r).__name__}"
                )

    def __call__(
        self, data: dict[str, Any],clinical,properties,num_examples: int = 10
    ) -> tuple[NdarrayOrTensor, NdarrayOrTensor, NdarrayOrTensor, float] | NdarrayOrTensor:
        """
        Args:
            data: dictionary data to be processed.
            num_examples: number of realizations to be processed and results combined.

        Returns:
            - if `return_full_data==False`: mode, mean, std, vvc. The mode, mean and standard deviation are
                calculated across `num_examples` outputs at each voxel. The volume variation coefficient (VVC)
                is `std/mean` across the whole output, including `num_examples`. See original paper for clarification.
            - if `return_full_data==False`: data is returned as-is after applying the `inferrer_fn` and then
                concatenating across the first dimension containing `num_examples`. This allows the user to perform
                their own analysis if desired.
        """
        d = dict(data)

        # check num examples is multiple of batch size
        if num_examples % self.batch_size != 0:
            raise ValueError("num_examples should be multiple of batch size.")

        # generate batch of data of size == batch_size, dataset and dataloader
        data_in = [deepcopy(d) for _ in range(num_examples)]
        ds = Dataset(data_in, self.transform)
        dl = DataLoader(ds, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=pad_list_data_collate)

        outs: list = []
        summ=0
        for checkpoint_path_tupl in self.checkpoint_paths:
            torch.cuda.empty_cache()

            is_classic_nnunet,checkpoint_path, arch_weight =checkpoint_path_tupl
            is_swin_monai=not is_classic_nnunet
            trainer= Main_trainer_pl(plans=self.plans,configuration=self.configuration
                            ,fold=self.fold ,dataset_json=self.dataset_json)
            trainer.on_train_start(is_swin_monai=is_swin_monai,is_classic_nnunet=is_classic_nnunet,checkpoint_path=checkpoint_path)
            pl_model=trainer.pl_model
            network=pl_model.network.cuda()
            network.eval()
            label_manager = trainer.plans_manager.get_label_manager(self.dataset_json)
            torch.cuda.empty_cache()
            for b in tqdm(dl) if has_tqdm and self.progress else dl:
            # do model forward pass

                # b[self._pred_key] = self.inferrer_fn(b[self.image_key].to(self.device))
                # curr=torch.sigmoid(execute_infrence(b[self.image_key],clinical,network,is_swin_monai))[0,:,:,:,:].detach().cpu().numpy()
                curr=torch.sigmoid(execute_infrence(b[self.image_key],clinical,network,is_swin_monai)).detach().cpu().numpy()*arch_weight
                summ=summ+arch_weight
                b[self._pred_key]=torch.tensor(curr)

                outs.extend([self.inverter(PadListDataCollate.inverse(i))[self._pred_key] for i in decollate_batch(b)])

        output: NdarrayOrTensor = stack(outs, 0)

        if self.return_full_data:
            return output

        # calculate metrics
        _mode = mode(output, dim=0)
        mean = output.sum(0)/summ
        std = output.std(0)
        vvc = (output.std() / output.mean()).item()
        del output
        arrs=list(map( lambda curr: my_convert_predicted_logits_to_segmentation_with_correct_shape(curr,
                                                                            plans_manager= trainer.plans_manager,
                                                                            configuration_manager= trainer.configuration_manager,
                                                                            label_manager=label_manager,
                                                                            properties_dict=properties,
                                                                            return_probabilities = True,
                                                                            num_threads_torch = 1),[_mode,mean,std]))
        return arrs[0], arrs[1], arrs[2], vvc




def test_time_augmentation(data
                           ,clinical
                           ,is_swin_monai
                           ,plans
                        ,configuration
                        ,fold,dataset_json
                        ,checkpoint_paths
                        ,properties
                        ,hparam_dict):
    keys = ["image"]
    sizee=(48, 192, 160)
    if(is_swin_monai):
        sizee=(64, 192, 160)
    is_classic_nnunet=not is_swin_monai



    val_transforms = Compose(
        [
            # RandAffined(
            #     keys,
            #     prob=1.0,
            #     spatial_size=sizee,
            #     rotate_range=(hparam_dict["rotate_a"], hparam_dict["rotate_b"], hparam_dict["rotate_c"]),
            #     # shear_range=(hparam_dict["shear_a"],hparam_dict["shear_b"],hparam_dict["shear_c"]),
            #     # translate_range=(0.1, 0.1, 0.1),
            #     # scale_range=((hparam_dict["scale_range_low"],hparam_dict["scale_range_high"])
            #     #              ,(hparam_dict["scale_range_low"],hparam_dict["scale_range_high"])
            #     #              ,(hparam_dict["scale_range_low"],hparam_dict["scale_range_high"])),
            #     padding_mode="zeros",
            #     mode=("bilinear"),
            # ),
            # CropForegroundd(keys, source_key="image"),
            # DivisiblePadd(keys, 16),

            # ScaleIntensityd("image"),
            # AdjustContrastd("image",hparam_dict["AdjustContrastd"]),
            Rand3DElasticd("image",sigma_range=(hparam_dict["sigma_low"],hparam_dict["sigma_low"]+hparam_dict["sigma_diff"])
                           , magnitude_range=(hparam_dict["magnitude_range_low"],hparam_dict["magnitude_range_low"]+hparam_dict["magnitude_range_diff"])
                           ,prob=hparam_dict["prob_elastic"]) 

        ]
    )
    # print(f"ddddddddd {data.shape}")
    data=data[0,:,:,:,:]
    tt_aug = My_TestTimeAugmentation(  val_transforms, batch_size=1, num_workers=0,plans=plans,configuration=configuration,fold=fold ,dataset_json=dataset_json
                                     ,is_swin_monai=is_swin_monai,is_classic_nnunet=is_classic_nnunet,checkpoint_paths=checkpoint_paths)
    mode_tta, mean_tta, std_tta, vvc_tta = tt_aug({"image" : data},clinical=clinical,properties=properties, num_examples=hparam_dict["num_examples"])

    # print(f"mmmm mean_tta {mean_tta.shape}")
    return mode_tta, mean_tta, std_tta, vvc_tta




import cupy


def save_target(arr,orig_im,name,temp_dir,physical_size):
    image= get_im_from_array(np.expand_dims(arr,0).astype(np.uint8),0,orig_im)  
    image=sitk.DICOMOrient(image, 'LPS')
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)) 
    # image=sitk.Resample(image, orig_im, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0)
    image=my_center_crop(image=image, physical_size=physical_size)
    writer = sitk.ImageFileWriter()
    newPath=f"{temp_dir}/{name}.nii.gz"
    writer.SetFileName(newPath)
    writer.Execute(image)
    return newPath

  

def case_preprocessing(plans_file,dataset_json_file,configuration, input_images_paths,temp_dir,anatomic_cols_paths):

    for p in anatomic_cols_paths:
        if len(p)<4 :
            return " "," "," "," "
    #load orient
    input_images= list(map(orientt,input_images_paths))
    #resampling to t2w
    input_images[1]=sitk.Resample(input_images[1], input_images[0], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)
    input_images[2]=sitk.Resample(input_images[2], input_images[0], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)

    #cropping
    physical_size=[138.0, 128.0, 124.0]
    # size=(48, 193, 165)
    # input_images=list(map(lambda im: crop_or_pad(image=im, size=size),input_images)) 
    t2w=input_images[0]
    
    pz=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[0],t2w),get_bool_arr_from_path(anatomic_cols_paths[1],t2w))
    full_pros=get_bool_arr_from_path(anatomic_cols_paths[4],t2w)
    # tz is rest of prostate not pz
    tz=np.logical_and(np.logical_not(pz),full_pros)
    #sv jointly
    sv=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[2],t2w),get_bool_arr_from_path(anatomic_cols_paths[3],t2w))




    input_images=list(map(lambda im: my_center_crop(image=im, physical_size=physical_size),input_images))

    torch.cuda.empty_cache()


    target_paths=[save_target(pz,t2w,"target_pz",temp_dir,physical_size)
                  ,save_target(tz,t2w,"target_tz",temp_dir,physical_size)
                  ,save_target(sv,t2w,"target_sv",temp_dir,physical_size)
                  ,save_target(full_pros,t2w,"target_full_pros",temp_dir,physical_size)
                  ]


    input_images_paths=list(map( lambda tupl: save_into_temp(tupl,temp_dir),list(zip(input_images,input_images_paths))))

    #bias field correction
    input_images=bias_field_and_normalize(input_images[0],input_images[1],input_images[2])
    cupy._default_memory_pool.free_all_blocks()

    #save back into files into temporary directory

    # temp_dir = "/workspaces/konwersjaJsonData/data/curr"



    pp = AnatomyPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images_paths, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)


    return data,properties,input_images_paths,target_paths

def get_el_1(arr):
    if(len(arr)>0):
        return arr[0]
    return " "
def get_bool_arr_from_path(pathh,ref_image):
    """    
    given path reads it resamples it to the space of reference  and return associated array
    then it casts it to boolean data type
    """
    # print(f"ppppp pathh {pathh} ref_image {ref_image} ")
    # ref_image=sitk.ReadImage(reference)
    imageA=sitk.ReadImage(pathh)

    imageA=sitk.Resample(imageA, ref_image, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0)
    return sitk.GetArrayFromImage(imageA).astype(bool)
    # return imageA#sitk.GetArrayFromImage(imageA).astype(bool)



def get_Metrics(one_hot,target,name=""):
    """
    return the metrics used in anatomy 
    """
    
    labelPred=sitk.GetImageFromArray(one_hot.astype(np.uint8))
    labelTrue=sitk.GetImageFromArray(target.astype(np.uint8))
    quality=dict()
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality[f"dice_{name}"]=dicecomputer.GetDiceCoefficient()
    quality[f"volume_similarity_{name}"]=dicecomputer.GetVolumeSimilarity()

    if(np.sum(one_hot.flatten())==0 or np.sum(target.flatten())==0):
        return list(quality.items()),quality[f"dice_{name}"]

    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality[f"avgHausdorff_{name}"]=hausdorffcomputer.GetAverageHausdorffDistance()
    # quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    return list(quality.items()),quality[f"dice_{name}"]


def full_infer_anatomy_case(plans_file,dataset_json_file,configuration, groupp,hparam_dict,is_swin_monai,checkpoint_paths,df):

    pat_num,group_dict=groupp

    our_prost_rows= df.iterrows()
    our_prost_rows= list(map(lambda tupl:tupl[1],our_prost_rows))
    row= list(filter(lambda roww : roww['patient_id']== pat_num,our_prost_rows))
    if(len(row)==0):            
        # print(f"eeeeeeeeee id {self.get_id_from_file_name(entry['data_file'])} file {entry['data_file']}  lll {len(row)}")
        row= [{'dre_result':-1.0,'patient_age':-1.0, 'psa_result' :-1.0 }]
    clinical = list(map(lambda row : np.array([my_to_float(row['dre_result']),my_to_float(row['patient_age']),my_to_float(row['psa_result'])]), row))
    clinical= np.stack(clinical)

    clinical=np.nan_to_num(clinical, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)
    clinical=torch.tensor(clinical)

    anatomic_cols=['cz_noSeg','pz_noSeg','sv_l_noSeg','sv_r_noSeg','pg_noSeg']
    input_names=["t2w","adc","hbv"]
    # print(group_dict)
    mris=list(map(lambda inner_dict: inner_dict[1]['series_MRI_path'], group_dict))
    input_paths=list(map(lambda name: list(filter(lambda el: name in el,mris)),input_names))
    input_paths=list(map(get_el_1,input_paths))
    anatomic_cols_paths=list(map( lambda col_name: list(map(lambda inner_dict: inner_dict[1][col_name], group_dict)),anatomic_cols))
    anatomic_cols_paths=list(map(lambda inner_list :list(filter(lambda el: len(el)>4,inner_list )) ,anatomic_cols_paths))
    if(len(anatomic_cols_paths[-1])==0 or len(anatomic_cols_paths[0])==0 or len(anatomic_cols_paths[1])==0 ):
        return " "
    anatomic_cols_paths= list(map(get_el_1,anatomic_cols_paths))



    # print(input_paths)
    # print(anatomic_cols_paths)

    

    temp_dir ="/workspaces/konwersjaJsonData/data/curr" 
    # temp_dir =tempfile.mkdtemp()# "/workspaces/konwersjaJsonData/data/curr" 
    data,properties,input_images_paths,target_paths=case_preprocessing(plans_file,dataset_json_file,configuration, input_paths,temp_dir,anatomic_cols_paths)
    if data==" ":
        return " "
    # size=(3,48, 192, 160)
    # data= crop_or_pad(image=data, size=size)
    data=einops.rearrange(data,'c z y x->1 c z y x')
    fold=0
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    mode_tta, mean_tta, std_tta, vvc_tta=test_time_augmentation(data
                           ,clinical
                           ,is_swin_monai
                           ,plans
                        ,configuration
                        ,fold,dataset_json
                        ,checkpoint_paths
                        ,properties
                        ,hparam_dict)

    #clearing temporary directory
    
    print(f"hhhhhhhhhh mean_tta {np.min(mean_tta.flatten())} {np.mean(mean_tta.flatten())} {np.max(mean_tta.flatten())}  ")
    print(f"hhhhhhhhhh mode_tta {np.min(mode_tta.flatten())} {np.mean(mode_tta.flatten())} {np.max(mode_tta.flatten())}  ")
    print(f"hhhhhhhhhh std_tta {np.min(std_tta.flatten())} {np.mean(std_tta.flatten())} {np.max(std_tta.flatten())}  ")

    #thresholding and getting single largest component
    mean_tta=(mean_tta>hparam_dict["treshold"]).astype(np.uint8)
    mean_tta=list(map( lambda i: get_largest_connected_component(mean_tta[i,:,:,:]),list(range(mean_tta.shape[0]))))
    mean_tta= np.stack(mean_tta)




    path_of_example=input_images_paths[0]  #"/workspaces/konwersjaJsonData/data/curr/1_t2w.nii.gz"

    #pz=pz+cz
    # pz=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[0],path_of_example),get_bool_arr_from_path(anatomic_cols_paths[1],path_of_example))
    # full_pros=get_bool_arr_from_path(anatomic_cols_paths[4],path_of_example)
    # # tz is rest of prostate not pz
    # tz=np.logical_and(np.logical_not(pz),full_pros)
    # #sv jointly
    # sv=np.logical_or(get_bool_arr_from_path(anatomic_cols_paths[2],path_of_example),get_bool_arr_from_path(anatomic_cols_paths[3],path_of_example))
    # krowa
    pz=sitk.GetArrayFromImage(sitk.ReadImage(target_paths[0])).astype(bool)
    

    # pz,full_pros,pz,sv    
    pz_metr=dict(get_Metrics(mean_tta[0,:,:,:],pz)[0])
    print(pz_metr)


    # tz_metr=get_Metrics(mean_tta[1,:,:,:],tz)[0]
    # sv_metr=get_Metrics(mean_tta[2,:,:,:],sv)[0]
    # full_metr=get_Metrics(mean_tta[3,:,:,:],full_pros)[0]

    # output=np.mean(np.stack(output),axis=0)
    # save_label(mode_tta,3,"mode_tta",path_of_example)
    save_label(mean_tta,0,"mean_pz",path_of_example,temp_dir)
    save_label(mean_tta,1,"mean_tz",path_of_example,temp_dir)
    save_label(mean_tta,2,"mean_sv",path_of_example,temp_dir)
    save_label(mean_tta,3,"mean_sum",path_of_example,temp_dir)


    # # save_label(np.expand_dims(pz,0).astype(np.uint8),0,"target_pz",path_of_example)  
    # # save_label(np.expand_dims(tz,0).astype(np.uint8),0,"target_tz",path_of_example)
    # # save_label(np.expand_dims(full_pros,0).astype(np.uint8),0,"target_full_pros",path_of_example)

    save_label(std_tta,0,"std_pz",path_of_example,temp_dir)    
    save_label(std_tta,1,"std_tz",path_of_example,temp_dir)    
    save_label(std_tta,2,"std_sv",path_of_example,temp_dir)    
    save_label(std_tta,3,"std_sum",path_of_example,temp_dir)    
    # shutil.rmtree(temp_dir, ignore_errors=True)

    return pz_metr#pz_metr["avgHausdorff_"]

# def objective(trial: optuna.trial.Trial,resCSVDir,test_ids_CSVDir,plans_file,dataset_json_file,configuration,comet_logger,df) -> float:
def objective(resCSVDir,test_ids_CSVDir,plans_file,dataset_json_file,configuration,comet_logger,df) -> float:
    hparam_dict={}
    # rotate=trial.suggest_float("rotate_a", 5.0,50.0)
    # hparam_dict["rotate_a"]=np.pi / rotate #np.pi / 10
    # hparam_dict["rotate_b"]=np.pi / rotate
    # hparam_dict["rotate_c"]=np.pi / trial.suggest_float("rotate_b", 5.0,50.0)
    # hparam_dict["shear_a"]=trial.suggest_float("shear_a", 0.0,10.0)
    # hparam_dict["shear_b"]=trial.suggest_float("shear_b", 0.0,10.0)
    # hparam_dict["shear_c"]=trial.suggest_float("shear_c", 0.0,10.0)
    # hparam_dict["scale_range_low"]=0.99
    # hparam_dict["scale_range_high"]=1.0

    # hparam_dict["AdjustContrastd"]=trial.suggest_float("AdjustContrastd", 0.0,10.0)#2
    hparam_dict["sigma_low"]=7.077968784031422#trial.suggest_float("sigma_low", 0.0,10.0)#5
    hparam_dict["sigma_diff"]=8.048997012706618#trial.suggest_float("sigma_diff", 0.0,10.0)#2
    hparam_dict["magnitude_range_low"]=3.85995070774654#trial.suggest_float("magnitude_range_low", 0.0,200.0)#50
    hparam_dict["magnitude_range_diff"]=172.39161544780146#trial.suggest_float("magnitude_range_diff", 0.0,400.0)#100


    hparam_dict["prob_elastic"]=1.0#trial.suggest_float("prob_elastic", 0.0,1.0)#1.0
    hparam_dict["num_examples"]=30#trial.suggest_int("num_examples", 8,16)
    hparam_dict["treshold"]=0.3#trial.suggest_float("treshold", 0.0,0.5)
    hparam_dict["swin_weight"]=0.9#trial.suggest_float("swin_weight", 0.0,1.0)



    checkpoint_paths=[(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_0/epoch=275-step=5796.ckpt",1.0)
                      ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_1/epoch=509-step=10710.ckpt",1.0)
                      ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_0/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_2/epoch=467-step=9828.ckpt",1.0)
                      ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_2/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=413-step=8694.ckpt",1.0)
                      ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_2/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=413-step=8694.ckpt",1.0)
                      ,(True,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/plain_3/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_4/epoch=449-step=9450.ckpt",1.0 )                     
                      
                      ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_0/epoch=359-step=7561.ckpt",hparam_dict["swin_weight"] )     
                      ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_1/epoch=539-step=11341.ckpt",hparam_dict["swin_weight"] )     
                      ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_2/epoch=395-step=8317.ckpt",hparam_dict["swin_weight"] )     
                      ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_3/epoch=323-step=6805.ckpt",hparam_dict["swin_weight"] )     
                      ,(False,"/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/fold_4/epoch=341-step=7183.ckpt",hparam_dict["swin_weight"] )     

                                                      ]
    
    is_swin_monai=False
    is_classic_nnunet=True



    #is_classic_nnunet,checkpoint_path

   
    test_ids=pd.read_csv(test_ids_CSVDir)['ids'].to_numpy().flatten()
    sourceFrame = pd.read_csv(resCSVDir) 
    grouped_rows=list(groupByMaster(list(sourceFrame.iterrows())))

    grouped_rows=list(filter(lambda groupp:groupp[0] not in test_ids,grouped_rows ))
    grouped_rows=grouped_rows[0:2]#TODO increase


    # print(grouped_rows[0])


    res=list(map( lambda groupp :full_infer_anatomy_case(plans_file,dataset_json_file,configuration, groupp,hparam_dict,is_swin_monai,checkpoint_paths,df), grouped_rows))
    res=list(filter(lambda el: el!=" ",res))
    print(f"rrrrrrrrrr {res}")
    avgHausdorff_=100.0
    try:
        avgHausdorff_=np.mean(list(map(lambda dd: dd["avgHausdorff_"], res)))
        dice_=np.mean(list(map(lambda dd: dd["dice_"], res)))
        volume_similarity_=np.mean(list(map(lambda dd: dd["volume_similarity_"], res)))
    except:    
        dice_=0.0
        volume_similarity_=0.0
        avgHausdorff_=100.0
    comet_logger.log_hyperparams(hparam_dict)
    comet_logger.log_metrics({"avgHausdorff_":avgHausdorff_,"dice_":dice_,"volume_similarity_":volume_similarity_  })
    

    return avgHausdorff_



if __name__ == '__main__':

    resCSVDir='/home/sliceruser/workspaces/konwersjaJsonData/outCsv/resCSV.csv'

    test_ids_CSVDir='/workspaces/konwersjaJsonData/test_ids.csv'
    # (paths to files may need adaptations)
    plans_file = '/workspaces/konwersjaJsonData/infrence/plans/anatomy_plans.json'
    dataset_json_file = '/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/dataset.json'
    configuration = '3d_lowres'
    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name="anatomy_infrence", # Optional
        #experiment_name="baseline" # Optional
    )    

    df=pd.read_csv("/workspaces/konwersjaJsonData/CRF.csv")
    df=df[['patient_id','dre_result','patient_age','psa_result']].replace("Dodatni (+)", "1.0")
    df=df[['patient_id','dre_result','patient_age','psa_result']].replace("Ujemny (-)", "0.0")
    df['dre_result']=pd.to_numeric(df['dre_result'])
    df['dre_result']=np.nan_to_num(df['dre_result'].to_numpy(),-1)

    objective(resCSVDir,test_ids_CSVDir,plans_file,dataset_json_file,configuration,comet_logger,df)
    # experiment_name="anatomy_infrence_l"
    # study = optuna.create_study(
    #         study_name=experiment_name
    #         ,sampler=optuna.samplers.CmaEsSampler()    
    #         # ,sampler=optuna.samplers.NSGAIISampler()    
    #         ,pruner=optuna.pruners.HyperbandPruner()
    #         # ,storage=f"mysql://root:jm@34.90.134.17:3306/{experiment_name}"
    #         ,storage=f"mysql://root@34.90.134.17/{experiment_name}"
    #         ,load_if_exists=True
    #         ,direction="minimize"
    #         )

    
    # #         #mysql://root@localhost/example
    # objective_p=partial(objective,resCSVDir=resCSVDir,test_ids_CSVDir=test_ids_CSVDir,plans_file=plans_file,dataset_json_file=dataset_json_file,configuration=configuration
    #                     ,comet_logger=comet_logger,df=df)

    # study.optimize(objective_p, n_trials=900)



# optuna-dashboard mysql://root@34.90.134.17/anatomy_infrence_j
# cd /workspaces/konwersjaJsonData
# python3 -m infrence.prprocess
