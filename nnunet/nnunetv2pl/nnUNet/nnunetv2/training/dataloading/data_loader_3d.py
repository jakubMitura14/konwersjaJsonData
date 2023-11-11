import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
import torch

# def transform_gold(target,seg_shape):
#     # shape_0 =target[0].shape
#     # seg_shape=shape_0
#     # shape_1 =(seg_shape[0],seg_shape[1] ,seg_shape[2]//2,seg_shape[3]//2,seg_shape[4]//2)
#     # shape_2 =(seg_shape[0],seg_shape[1] ,shape_1[2]//2,shape_1[3]//2,shape_1[4]//2)
#     # shape_3 =(seg_shape[0],seg_shape[1] ,shape_2[2]//2,shape_2[3]//2,shape_2[4]//2)
    
#     shape_1 =(seg_shape[2]//2,seg_shape[3]//2,seg_shape[4]//2)
#     shape_2 =(shape_1[0]//2,shape_1[1]//2,shape_1[2]//2)
#     shape_3 =(shape_2[0]//2,shape_2[1]//2,shape_2[2]//2)
#     shape_4 =(shape_3[0]//2,shape_3[1]//2,shape_3[2]//2)
    
#     targets=[target[0]]
#     shapes=[seg_shape,shape_1,shape_2,shape_3,shape_4]     
#     for j in range():
#         if(j>0):
#             tensor=torch.unsqueeze(torch.tensor(target[j]),dim=0)
#             # print(f"fffff {tensor.shape}  shapes[j] {shapes[j]}")
#             interp=torch.nn.functional.interpolate(input=tensor.float(),size=shapes[j],mode='nearest-exact')
#             print(f"uuu {interp.shape}")
#             loc_res=interp.numpy().astype(np.short)[0,:,:,:,:]
#             targets.append(loc_res)
#     return targets

def my_to_float(strr):
    if(type(strr) is float):
        return strr
    strr= strr.replace(",",".")
    return float(strr)
    

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        # print(f"sssssself.seg_shape  {self.seg_shape} ")
        


        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties,rows = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

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
            clinical = list(map(lambda row : np.array([my_to_float(row['dre_result']),my_to_float(row['patient_age']),my_to_float(row['psa_result'])]), rows))
            clinical= np.stack(clinical)
            clinical=np.nan_to_num(clinical, copy=True, nan=-1.0, posinf=-1.0, neginf=-1.0)

            # seg_all=transform_gold(seg_all,self.seg_shape)
        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys, 'clinical':clinical}


# if __name__ == '__main__':
#     folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
#     ds = nnUNetDataset(folder, 0)  # this should not load the properties!
#     dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
#     a = next(dl)
