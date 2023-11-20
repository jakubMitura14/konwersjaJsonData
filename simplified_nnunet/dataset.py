import os
from typing import List
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
import pandas as pd
from batchgenerators.dataloading.data_loader import DataLoader
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import multiprocessing
import os
from multiprocessing import Pool
from typing import List
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from typing import Union, Tuple, List
import einops
import pandas as pd
from .transforms import get_training_transforms,get_validation_transforms


class nnUNetDataset_custom(object):
    """
    dataset_ids - ids used for either training or validation depending on usage
    """
    def __init__(self,dataset_ids:np.ndarray,hdf5_file,plans_file,dataset_json,input_channels,is_lesion_segm=True,is_anatomy_segm=False
                 ,is_training=True):
        super().__init__()
        configuration="3d_lowres"
        self.dataset_ids = dataset_ids
        self.hdf5_file = hdf5_file
        #hyperparameter set as is in nnunet
        self.oversample_foreground_percent = 0.33
        self.plans_manager = PlansManager(plans_file)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.batch_size = self.configuration_manager.batch_size
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.rotation_for_DA=rotation_for_DA
        self.do_dummy_2d_data_aug=do_dummy_2d_data_aug
        self.initial_patch_size=initial_patch_size
        self.patch_size=self.initial_patch_size
        self.mirror_axes=mirror_axes
        final_patch_size=self.configuration_manager.patch_size
        self.need_to_pad = (np.array(self.initial_patch_size) - np.array(final_patch_size)).astype(int)
        self.data_shape, self.seg_shape = self.determine_shapes(input_channels)
        self.has_ignore=False
        self.annotated_classes_key = tuple(self.label_manager.all_labels)
        self.probabilistic_oversampling=False
        self.get_do_oversample = self._oversample_last_XX_percent if not self.probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.deep_supervision_scales=self._get_deep_supervision_scales()
        if(is_training):
            self.transforms= get_training_transforms(patch_size=self.patch_size
                                            , rotation_for_DA=self.rotation_for_DA
                                            , do_dummy_2d_data_aug=self.do_dummy_2d_data_aug
                                            , mirror_axes=self.mirror_axes
                                            , deep_supervision_scales=self.deep_supervision_scales
                                            ,is_lesion_segm=is_lesion_segm
                                            , is_anatomy_segm=is_anatomy_segm)
        else:
            self.transforms = get_validation_transforms(deep_supervision_scales=self.deep_supervision_scales)    

    def _get_deep_supervision_scales(self):
        deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
            self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        return deep_supervision_scales


    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self,input_channels):
        # load one case


        data_shape = (self.batch_size, input_channels, *self.patch_size)
        seg_shape = (self.batch_size, 1, *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(self.data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + self.data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - self.data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [self.data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


    def __getitem__(self, key):
        gr = self.hdf5_file[str(key)]        
        data = gr['mri_data'][:,:,:,:]
        pz=gr["tresh_pz"][:,:,:]
        tz=gr["tresh_tz"][:,:,:]
        anat= np.stack((pz,tz),axis=0)
        data= np.concatenate((data,anat),axis=0)
        target = gr['all_lesions'][:,:,:] 
        data = einops.rearrange(data,"c h w d -> 1 c h w d")       
        seg = einops.rearrange(target,"h w d -> 1 1 h w d")       
        clinical = np.array([gr.attrs['dre'], gr.attrs['age'], gr.attrs['psa'], gr.attrs['id']])
        dictt=self.transforms (**{"data":data,"seg" :target})

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        selected_keys = self.get_indices()

        shape = data_all.shape[1:]
        dim = 3
      
        data=dictt["data"]
        seg=dictt["target"]
        for j, i in enumerate(selected_keys):

            force_fg = self.get_do_oversample(j)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, None)
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)


        return  data_all,seg_all,clinical

    def __len__(self):
        return len(self.dataset_ids)


        # return data, seg, entry['properties'],row

#  nnUNetDataLoader3D(dataset_tr, self.batch_size,
#                                        initial_patch_size,
#                                        self.configuration_manager.patch_size,
#                                        self.label_manager,
#                                        oversample_foreground_percent=self.oversample_foreground_percent,
#                                        sampling_probabilities=None, pad_sides=None)