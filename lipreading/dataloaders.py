import numpy as np
import torch

from lipreading.dataset import MyDataset, pad_packed_collate
from lipreading.preprocess import *


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        preprocessing['train'] = Compose([
                                    # RgbToGray(), # if image is RGB
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])

        preprocessing['val'] = Compose([
                                    # RgbToGray(), # if image is RGB
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])

        preprocessing['test'] = preprocessing['val']

    elif modality == 'raw_audio':

        preprocessing['train'] = Compose([
                                    AddNoise( noise=np.load('./data/babbleNoise_resample_16K.npy')),
                                    NormalizeUtterance()])

        preprocessing['val'] = NormalizeUtterance()

        preprocessing['test'] = NormalizeUtterance()

    return preprocessing


def get_data_loaders(args, partition):
    preprocessing = get_preprocessing_pipelines(args.modality)

    # create dataset object for each partition
    dsets = {p: MyDataset(
                modality=args.modality,
                data_partition=p,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                is_var_length = args.var_length,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[p],
                data_suffix='.npz'
                ) for p in partition}
    
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partition}
    return dset_loaders

