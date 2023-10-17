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
import multiprocessing
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




def input_verification_crop_or_pad(
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Optional[Iterable[int]] = (20, 256, 256),
    physical_size: Optional[Iterable[float]] = None,
) -> Tuple[Iterable[int], Iterable[int]]:
    """
    Calculate target size for cropping and/or padding input image

    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
    - shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    - size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        spacing_zyx = list(image.GetSpacing())[::-1]
        size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
        size_zyx = [int(np.round(x)) for x in size_zyx]

        if size is None:
            # use physical size
            size = size_zyx
        else:
            # verify size
            if size != size_zyx:
                raise ValueError(f"Size and physical size do not match. Size: {size}, physical size: "
                                 f"{physical_size}, spacing: {spacing_zyx}")

    if isinstance(image, sitk.Image):
        # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
        shape = image.GetSize()
        size = list(size)[::-1]
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size)
    rank = len(size)
    assert rank <= len(shape) <= rank + 1, \
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

    return shape, size


def crop_or_pad(
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Optional[Iterable[int]] =None,# (20, 256, 256),
    physical_size: Optional[Iterable[float]] = None,
    crop_only: bool = True,
) -> "Union[sitk.Image, npt.NDArray[Any]]":
    """
    copied from picai_prep

    Resize image by cropping and/or padding

    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
    - resized image (same type as input)
    """
    # input conversion and verification
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    rank = len(size)
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if shape[i] < size[i]:
            if crop_only:
                continue

            # set padding settings
            padding[i][0] = (size[i] - shape[i]) // 2
            padding[i][1] = size[i] - shape[i] - padding[i][0]
        else:
            # create slicer object to crop image
            idx_start = int(np.floor((shape[i] - size[i]) / 2.))
            idx_end = idx_start + size[i]
            slicer[i] = slice(idx_start, idx_end)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image[tuple(slicer)])
    else:
        return np.pad(image[tuple(slicer)], padding)




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
        
            
### bias field correction and normalization
#on the basis of https://github.com/lucianoAvinas/lapgm/blob/main/examples/image_correction.ipynb
def bias_field_and_normalize_help(t2w_image,adc_image,hbv_image):
    # Approximate location of farthest peak for true data.
    # In practice this can be set to any fixed value of choice.
    TRGT = 1.0
    #first bias field correction
    modalities_to_normalize=  [t2w_image,adc_image,hbv_image]
    modalities_to_normalize = list(map(sitk.GetArrayFromImage ,modalities_to_normalize))
    arrrr = lapgm.to_sequence_array(modalities_to_normalize)
    params = get_paramss(arrrr)
    # Run debias procedure and take parameter output
    arrrr= lapgm.debias(arrrr, params)
    arrrr_debiased=arrrr.copy()
    nan_summ=np.sum(np.isnan(arrrr))
    if(nan_summ>0):
        print(f"eeeeeeeeeeeeeerrorrr naaans")
        raise Exception('nanns')   
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


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/workspaces/konwersjaJsonData/infrence/plans/anatomy_plans.json'
    dataset_json_file = '/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/dataset.json'
    configuration = '3d_lowres'


    #'t2w','adc','hbv'
    main_dat='/home/sliceruser/workspaces/konwersjaJsonData/AI4AR_cont/Data/001'

    input_images_paths = [f"{main_dat}/1_t2w.mha",f"{main_dat}/1_adc.mha",f"{main_dat}/1_hbv.mha" , ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']
    #load orient
    input_images= list(map(orientt,input_images_paths))
    #resampling to t2w
    input_images[1]=sitk.Resample(input_images[1], input_images[0], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)
    input_images[2]=sitk.Resample(input_images[2], input_images[0], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkBSpline, 0)

    #cropping
    physical_size=[138.0, 128.0, 124.0]
    input_images=list(map(lambda im: crop_or_pad(image=im, physical_size=physical_size),input_images))

    #bias field correction
    input_images=bias_field_and_normalize(input_images[0],input_images[1],input_images[2])


    #save back into files into temporary directory
    # temp_dir = tempfile.mkdtemp() 
    temp_dir = "/workspaces/konwersjaJsonData/explore/temp"
    input_images_paths=list(map( lambda tupl: save_into_temp(tupl,temp_dir),list(zip(input_images,input_images_paths))))

    pp = AnatomyPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images_paths, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    #clearing temporary directory
    # shutil.rmtree(temp_dir, ignore_errors=True)
   
    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    # data=example_test_case_preprocessing()
    # print(f"ddddd {data.shape}")
    plans_file = '/workspaces/konwersjaJsonData/infrence/plans/anatomy_plans.json'
    dataset_json_file = '/workspaces/konwersjaJsonData/data/anatomy_res/nnunet_classic/swin_all/results_out/Main_trainer_pl__nnUNetPlans__3d_lowres/dataset.json'
    configuration = '3d_lowres'
    fold=0
    
    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    Main_trainer_pl(plans=plans,configuration=configuration
                    ,fold=fold ,dataset_json=dataset_json)

 
        # self.network = self.network.to(self.device)


#python3 -m infrence.prprocess


#python3 -m infrence.anatomy_model.nnUNetTrainer.Main_trainer_pl

    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
