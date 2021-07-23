"""Functions for reading CHIP34140702 data director."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import os.path
import pickle

import cv2
import matplotlib.pyplot
from imageio import imwrite
from scipy.ndimage import rotate

from utils.MINC import *
from utils.image_utils import crop, crop_center
from utils.tfrecord_utils import *


class CHIP34140702(object):
    FILTER_TYPES = ['OK', 'NG'] #['NORMAL', 'MILDMS', 'MODERATEMS', 'SEVEREMS']
    SET_TYPES = ['TRAIN', 'VAL', 'TEST']
    LABELS = {'Background': 0, 'Void': 1, 'Peeling': 2, 'Scratch': 3, 'Stain': 4, 'Crack': 5}
    
    class Options(object):
        def __init__(self):
            self.description = None
            self.dir = os.path.dirname(os.path.realpath(__file__))
            self.folderOK = 'OK' #folderNormal
            self.folderNG = 'NG' #folderMildMS, folderModerateMS, folderSevereMS
            self.folderGT = 'groundtruth'
            self.numSamples = -1
            self.partition = {'TRAIN': 0.6, 'VAL': 0.15, 'TEST': 0.25}
            self.sliceStart = 20
            self.sliceEnd = 140
            self.useCrops = False
            self.cropType = 'random'  # random or center
            self.numRandomCropsPerSlice = 5
            self.rotations = [0]
            self.cropWidth = 128
            self.cropHeight = 128
            self.cache = False
            self.sliceResolution = None  # format: HxW
            self.addInstanceNoise = False  # Affects only the batch sampling. If True, a tiny bit of noise will be added to every batch
            self.filterType = None  # MILDMS, MODERATEMS, SEVEREMS, NORMAL
            self.axis = 'axial'  # saggital, coronal or axial
            self.debug = False
            self.normalizationMethod = 'standardization'
            self.skullRemoval = False
            self.backgroundRemoval = False

    def __init__(self, options=Options()):
        self.options = options

        if options.cache and os.path.isfile(self.pckl_name()):
            f = open(self.pckl_name(), 'rb')
            tmp = pickle.load(f)
            f.close()
            self._epochs_completed = tmp._epochs_completed
            self._index_in_epoch = tmp._index_in_epoch
            self.cases = self._get_cases() # self._get_patients()
            self._images, self._labels, self._sets = read_tf_record(self.tfrecord_name())

            f = open(self.split_name(), 'rb')
            self.cases_split = pickle.load(f)
            f.close()
            if not os.path.exists(self.split_name() + ".deprecated"):
                os.rename(self.split_name(), self.split_name() + ".deprecated")
            self._convert_case_split()

            self._epochs_completed = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
            self._index_in_epoch = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}
        else:
            # Collect all patients
            self.cases = self._get_cases()
            self.cases_split = {}  # Here we will later store the info whether a patient belongs to train, val or test

            # Determine Train, Val & Test set based on patients
            if not os.path.isfile(self.split_name()):
                _num_cases = len(self.cases)
                _ridx = numpy.random.permutation(_num_cases)

                _already_taken = 0
                for split in self.options.partition.keys():
                    if 1.0 >= self.options.partition[split] > 0.0:
                        num_cases_for_current_split = max(1, math.floor(self.options.partition[split] * _num_cases))
                    else:
                        num_cases_for_current_split = int(self.options.partition[split])

                    if num_cases_for_current_split > (_num_cases - _already_taken):
                        num_cases_for_current_split = _num_cases - _already_taken

                    self.cases_split[split] = _ridx[_already_taken:_already_taken + num_cases_for_current_split]
                    _already_taken += num_cases_for_current_split

                self._convert_case_split()  # NEW! We have a new format for storing hte patientsSplit which is OS agnostic. #self._convert_patient_split()
            else:
                f = open(self.split_name(), 'rb')
                self.cases_split = pickle.load(f)
                f.close()
                self._convert_case_split()  # NEW! We have a new format for storing hte patientsSplit which is OS agnostic.

    def _get_cases(self): # 
        return CHIP34140702.get_cases(self.options)


    @staticmethod
    def get_cases(options): #get_patients
        minc_folders = [options.folderOK, options.folderNG]

        # Iterate over all folders and collect patients
        cases = []
        for n, minc_folder in enumerate(minc_folders):
            if minc_folder == options.folderOK:
                _type = 'OK'
            elif minc_folder == options.folderNG:
                _type = 'NG'

            # Continue with the next patient if the current one is not part of the desired types
            if _type not in options.filterType:
                continue

            _regex = "*.mnc.gz"
            _files = glob.glob(os.path.join(options.dir, minc_folder, _regex))
            for f, fname in enumerate(_files):
                case = {
                    'name': os.path.basename(fname),
                    'type': _type,
                    'fullpath': fname
                }
                case['filtered_files'] = case['fullpath']

                if case['type'] == 'OK':
                    case['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'ok.mnc.gz')
                elif case['type'] == 'NG':
                    patient['groundtruth_filename'] = os.path.join(options.dir, options.folderGT, 'ng.mnc.gz')

                cases.append(case)

        return cases


    @property
    def images(self):
        return self._images

    @property
    def num_channels(self):
        return self._images.shape[3]



    def name(self):
        _name = "CHIP34140702"
        if self.options.description:
            _name += "_{}".format(self.options.description)
        if self.options.numSamples > 0:
            _name += '_n{}'.format(self.options.numSamples)
        _name += "_p{}-{}-{}".format(self.options.partition['TRAIN'], self.options.partition['VAL'], self.options.partition['TEST'])
        if self.options.useCrops:
            _name += "_{}crops{}x{}".format(self.options.cropType, self.options.cropWidth, self.options.cropHeight)
            if self.options.cropType == "random":
                _name += "_{}cropsPerSlice".format(self.options.numRandomCropsPerSlice)
        if self.options.sliceResolution is not None:
            _name += "_res{}x{}".format(self.options.sliceResolution[0], self.options.sliceResolution[1])
        if self.options.skullRemoval:
            _name += "_noSkull"
        if self.options.backgroundRemoval:
            _name += "_noBackground"
        return _name

    def pckl_name(self):
        return os.path.join(self.dir(), self.name() + ".pckl")

    def tfrecord_name(self):
        return os.path.join(self.dir(), self.name() + ".tfrecord")

    def split_name(self):
        return os.path.join(self.dir(),'split-{}-{}-{}.pckl'.format(self.options.partition['TRAIN'], self.options.partition['VAL'], self.options.partition['TEST']))

    def dir(self):
        return self.options.dir


    def _convert_case_split(self): #_convert_patient_split
        for split in self.cases_split.keys():
            _list_of_case_names = []
            for pidx in self.cases_split[split]:
                if not isinstance(pidx, str):
                    _list_of_case_names += [self.cases[pidx]['name']]
                else:
                    _list_of_case_names = self.cases_split[split]
                    break
            self.cases_split[split] = _list_of_case_names

        f = open(self.split_name(), 'wb')
        pickle.dump(self.cases_split, f)
        f.close()
