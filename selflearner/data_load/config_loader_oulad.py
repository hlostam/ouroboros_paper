import logging

from pandas import HDFStore

from selflearner.data_load.config_loader import ConfigLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigLoaderOulad(ConfigLoader):
    DEFAULT_DATA_HDF5_PATH = 'data/selflearner.h5'
    DEFAULT_OULAD_HDF5_PATH = 'data/oulad.h5'

    def __init__(self, problem_definition, data_hdf5_path = DEFAULT_DATA_HDF5_PATH, oulad_hdf5_path=DEFAULT_OULAD_HDF5_PATH):
        super().__init__(problem_definition)
        self.__oulad_hdf5_path = oulad_hdf5_path
        self.__data_hdf5_path = data_hdf5_path

    def load_config(self, force_remote=False):
        logging.debug("Loading config for OULAD")
        key = self.__get_config_key()
        with HDFStore(self.__data_hdf5_path) as store:
            if key not in store:
                raise Exception("Config not available")
                # logging.debug("Loading config from HDF5")
                # config =  self.__load_config_from_hdf5()
            else:
                config = store['key']
                return config

    def __get_config_key(self):
        if super().problem_definition.assessment_name:
            return "/".join([super().problem_definition.module, super().problem_definition.presentation, super().problem_definition.presentation_train,
                             "assesments", super().problem_definition.assessment_name, str(super().problem_definition.days_to_cutoff)])
        elif super().problem_definition.days_of_presentation:
            return "/".join([super().problem_definition.module, super().problem_definition.presentation, super().problem_definition.presentation_train,
                             "days", super().problem_definition.days_of_presentation])
        else:
            raise AttributeError("Neither assessment name nor days_to_presentation defined")

