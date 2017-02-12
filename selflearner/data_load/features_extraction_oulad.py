import hashlib
import logging
import os
import urllib
import zipfile
import math

import pandas as pd
from genericpath import isfile

import shutil
from os.path import join

import re

from selflearner.data_load.config_loader import Config
from selflearner.data_load.feature_extraction import FeatureExtraction
from selflearner.data_load.features_utils import FeaturesMapping

from selflearner.data_load.hdf5.pytables_descriptions import ConfigDescriptionOulad
from selflearner.data_load.hdf5.pytables_hdf5_manager import PytablesHdf5Manager
from selflearner.problem_definition import ProblemDefinition
from selflearner.problem_definition import TrainingType

pd.set_option('display.max_columns', 0)  # Display any number of columns
pd.set_option('display.max_rows', 0)  # Display any number of rows
pd.set_option('expand_frame_repr', False)

__author__ = 'author@gmail.com'

DATA_HDF5_PATH = 'data/selflearner.h5'
DEFAULT_HDF5_PATH = 'data/oulad.h5'
OULAD_URL = 'https://analyse.kmi.open.ac.uk/open_dataset/download'
OULAD_MD5_URL = ' https://analyse.kmi.open.ac.uk/open_dataset/downloadCheckSum'

DS_ASSESSMENTS = 'assessments'
DS_COURSES = 'courses'
DS_STUD_ASSESSMENTS = 'studentAssessment'
DS_STUD_INFO = 'studentInfo'
DS_STUD_REG = 'studentRegistration'
DS_STUD_VLE = 'studentVle'
DS_VLE = 'vle'

DS_ARRAY = [DS_ASSESSMENTS, DS_COURSES, DS_STUD_ASSESSMENTS, DS_STUD_INFO,
            DS_STUD_REG, DS_STUD_VLE, DS_VLE]

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name) - %(levelname)s - %(message)s')


class Hdf5Creator:
    def __init__(self, csv_path=None, hdf5_path=os.path.join(os.path.dirname(__file__), DEFAULT_HDF5_PATH)):
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.hdf5_manager = PytablesHdf5Manager(self.hdf5_path)
        self.logger = logging.getLogger(__name__)

    def create_hdf5(self, cleanup_csv=True):
        try:
            os.remove(self.hdf5_path)
            self.logger.debug("Removing previous hdf5file: %s", self.hdf5_path)
        except OSError:
            self.logger.error("Cannot close hdf5", exc_info=True)
            pass

        tmp_csv_path = "oulad_tmp"
        if self.csv_path is None:
            self.prepare_zip_and_extract(tmp_csv_path)
        else:
            tmp_csv_path = self.csv_path

        self.read_all_csv_to_hdf5(tmp_csv_path)

        if cleanup_csv:
            self.logger.debug("Deleting the tmp directory and all the files")
            shutil.rmtree(tmp_csv_path)

    def prepare_zip_and_extract(self, tmp_csv_path):
        tmp_zip_file = 'oulad.zip'
        self.logger.debug("Csv path is none, looking for zip file %s", tmp_zip_file)
        if not isfile(tmp_zip_file):
            self.logger.debug("Zip file not found -> downloading from: %s", OULAD_URL)
            self.download_oulad_dataset(tmp_zip_file)
        else:
            self.logger.debug("Existing zip found, checking MD5 checksum")
            is_md5_same = self.check_oulad_md5(tmp_zip_file)
            if not is_md5_same:
                self.logger.debug("MD5 of the current zip doesn't match, trying to download one more time")
                self.download_oulad_dataset(tmp_zip_file)
        archive = zipfile.ZipFile(tmp_zip_file, 'r')
        self.logger.debug("Extracting zip")
        archive.extractall(tmp_csv_path)

    def check_oulad_md5(self, tmp_zip_file):
        with urllib.request.urlopen(OULAD_MD5_URL) as response:
            data = response.read()
            md5_zip_desired = data.decode('UTF-8')
            self.logger.debug("Desired MD4: %s", md5_zip_desired)
        md5_zip_real = ""
        if isfile(tmp_zip_file):
            hasher = hashlib.md5()
            with open(tmp_zip_file, 'rb') as afile:
                buf = afile.read()
                hasher.update(buf)
            md5_zip_real = hasher.hexdigest()
            self.logger.debug("Real MD5: %s", md5_zip_real)
        return md5_zip_desired == md5_zip_real

    def download_oulad_dataset(self, tmp_zip_file):
        with urllib.request.urlopen(OULAD_URL) as response, open(tmp_zip_file, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        if not self.check_oulad_md5(tmp_zip_file):
            raise Exception('MD5 checksum of %s after download doesnt match the desired one', tmp_zip_file)

    def read_all_csv_to_hdf5(self, csv_path):
        # with HDFStore(self.hdf5_path) as store:
        self.logger.debug("Reading all csvs..")
        successfully_read = 0
        for ds in DS_ARRAY:
            successfully_read = self.read_csv_and_store_to_hdf5(csv_path, ds, successfully_read)
        if successfully_read != len(DS_ARRAY):
            self.logger.error(
                "Successfully read = " + str(successfully_read) + " <> number of expected files " + str(
                    len(DS_ARRAY)))
        self.logger.debug("Read: %s", str(successfully_read))

    def read_csv_and_store_to_hdf5(self, csv_path, ds, successfully_read):
        f = ".".join([ds, 'csv'])
        self.logger.debug("Reading: %s ", str(f))
        abs_path = join(csv_path, f)
        if isfile(abs_path):
            df = pd.read_csv(abs_path)
            df = self.preprocess(df, ds)
            # with HDFStore(self.hdf5_path) as store:
            #     store[ds]= df
            self.hdf5_manager.store_dataframe(ds, df)
            successfully_read += 1
        return successfully_read

    @staticmethod
    def preprocess(df, ds):
        if ds == DS_ASSESSMENTS:
            df = df.set_index(['code_module', 'code_presentation'], drop=False)
            df = df.sort_values(by='date')
            df_agg = df.groupby(['code_module', 'code_presentation', 'assessment_type'])
            # Assign each row the order within the group MOD-PRES-TYPE
            df['assessment_name'] = df['assessment_type'] + " " + (df_agg.cumcount() + 1).map(str)
            df.loc[df['assessment_name'] == 'Exam 1', 'assessment_name'] = 'Exam'
        elif ds == DS_COURSES:
            df = df.set_index(['code_module', 'code_presentation'])
        elif ds == DS_VLE:
            df = df.set_index(['code_module', 'code_presentation'])
        elif ds == DS_STUD_INFO:
            df = df.set_index(['code_module', 'code_presentation'])
        elif ds == DS_STUD_REG:
            df = df.set_index(['code_module', 'code_presentation'])
        elif ds == DS_STUD_ASSESSMENTS:
            df = df.set_index(['id_assessment'])
        elif ds == DS_STUD_VLE:
            df = df.set_index(['code_module', 'code_presentation'])
        else:
            df = df
        return df

    def get_hdf5_manager(self):
        self.check_hdf5()
        return self.hdf5_manager

    def check_hdf5(self, reload=False):
        if reload:
            self.create_hdf5()
        elif isfile(self.hdf5_path):
            if not PytablesHdf5Manager(self.hdf5_path).check_exist_dataframes(DS_ARRAY):
                self.create_hdf5()
        else:
            self.create_hdf5()


class Hdf5Features:
    DF_STR = 'df'
    CONFIG_STR = 'config'
    NP_STR = 'np'
    TRAIN_STR = 'train'
    TEST_STR = 'test_string'
    ASSESSMENTS_STR = 'assessments'
    DAYS_STR = 'days'

    def __init__(self, problem_definition, file_path=os.path.join(os.path.dirname(__file__), DATA_HDF5_PATH)):
        """

        :type file_path: String
        :type problem_definition: ProblemDefinition
        """
        self.file_path = file_path
        self.problem_definition = problem_definition
        self.access_str = problem_definition.string_representation
        self.storage_manager = PytablesHdf5Manager(file_path)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("TMA ACCESS STR: %s", self.access_str)
        self.logger.debug("Storage manager: %s", self.storage_manager)

    def __create_alias__(self, config):
        """
        Creates the symlink so that the problem might be referenced in two ways
         1] /MOD/PRES/
         :type config: Config
        """
        type = self.problem_definition.problem_definition_type
        if type == ProblemDefinition.MODPRES_DAY:
            alias = self.problem_definition.get_assessment_type_str(config.assessment_name, config.days_to_cutoff)
        else:
            alias = self.problem_definition.get_modpres_day_str(str(config.current_date_test))
        self.storage_manager.create_alias(alias, self.access_str)
        # Store the alias in HDF5
        # H5pyHdfsManager(self.file_path).create_symlink(alias, self.access_str)
        # # with h5py.File(self.file_path, 'w') as f:
        #     f[alias] = h5py.SoftLink('/' + self.access_str)
        self.logger.debug("Alias created: %s", alias)

    def store_config(self, config):
        config_access_str = '/'.join([self.access_str, self.CONFIG_STR])
        # with HDFStore(self.file_path) as store:
        #     store[config_access_str] = config
        # PandasHdf5Manager(self.file_path).store_object(config_access_str, config.to_dict())
        self.storage_manager.store_table_one_row(config, config_access_str, ConfigDescriptionOulad)
        self.__create_alias__(config)

    def load_config(self):
        """
        Tries to load the config from the HDF file, if the config is not in the file Exception is raised
        :return: Config file
        """
        config_access_str = '/'.join([self.access_str, self.CONFIG_STR])
        row = self.storage_manager.load_object_one_row(config_access_str)
        if row is not None:
            c = Config.from_pytable_row(row)
            return c
        else:
            return None

    def load_df(self, df_name):
        df_access_str = '/'.join([self.access_str, self.DF_STR, df_name])
        return self.storage_manager.load_dataframe(df_access_str)

    def store_df(self, df_name, df):
        """
        Stores the dataframe in the HDF5 file with the given name. The name shoudl follow the conventions.
        :param df_name:
        :param df:
        """
        df_access_str = '/'.join([self.access_str, self.DF_STR, df_name])
        # with HDFStore(self.file_path) as store:
        #     store[df_access_str] = df
        self.storage_manager.store_dataframe(df_access_str, df)


class FeatureExtractionOulad(FeatureExtraction):
    def prepare_student_data(self, source_type):
        pass

    data_directory = "data"

    def __init__(self, problem_definition: ProblemDefinition,
                 hdf5_path=os.path.join(os.path.dirname(__file__), DEFAULT_HDF5_PATH),
                 include_submitted=False):
        super().__init__(problem_definition, include_submitted=include_submitted)
        self.hdf5_path = hdf5_path
        self.data_hdf5_manager = Hdf5Features(problem_definition)
        self.store_manager = PytablesHdf5Manager(hdf5_path)
        self.logger = logging.getLogger(__name__)

    def prepare_df_by_modpres(self):
        id_assessment_train = self.__config.id_assessment_train
        id_assessment_test = self.__config.id_assessment_test

        df_assessments = self.store_manager.load_dataframe(DS_ASSESSMENTS).sort_index()
        self.dfs_train = {
            DS_COURSES: self.store_manager.load_dataframe(DS_COURSES).ix[self.module, self.presentation_train],
            DS_ASSESSMENTS: df_assessments.ix[self.module, self.presentation_train],
            DS_VLE: self.store_manager.load_dataframe(DS_VLE).ix[self.module, self.presentation_train],
            DS_STUD_INFO: self.store_manager.load_dataframe(DS_STUD_INFO).ix[self.module, self.presentation_train],
            DS_STUD_ASSESSMENTS: self.store_manager.load_dataframe(DS_STUD_ASSESSMENTS).ix[id_assessment_train],
            DS_STUD_REG: self.store_manager.load_dataframe(DS_STUD_REG).ix[self.module, self.presentation_train],
            DS_STUD_VLE: self.store_manager.load_dataframe(DS_STUD_VLE).ix[self.module, self.presentation_train]
        }
        self.dfs_test = {
            DS_COURSES: self.store_manager.load_dataframe(DS_COURSES).ix[self.module, self.presentation],
            DS_ASSESSMENTS: df_assessments.ix[self.module, self.presentation],
            DS_VLE: self.store_manager.load_dataframe(DS_VLE).ix[self.module, self.presentation],
            DS_STUD_INFO: self.store_manager.load_dataframe(DS_STUD_INFO).ix[self.module, self.presentation],
            DS_STUD_ASSESSMENTS: self.store_manager.load_dataframe(DS_STUD_ASSESSMENTS).ix[id_assessment_test],
            DS_STUD_REG: self.store_manager.load_dataframe(DS_STUD_REG).ix[self.module, self.presentation],
            DS_STUD_VLE: self.store_manager.load_dataframe(DS_STUD_VLE).ix[self.module, self.presentation]
        }

    def check_hdf5(self, reload=False):
        if reload:
            Hdf5Creator().create_hdf5()
        elif isfile(self.hdf5_path):
            if not PytablesHdf5Manager(self.hdf5_path).check_exist_dataframes(DS_ARRAY):
                Hdf5Creator().create_hdf5()
        else:
            Hdf5Creator().create_hdf5()

    def retrieve_filtered_students(self, dfs):
        df_studinfo = dfs[DS_STUD_INFO]

        # Filter banked students
        df_ass = dfs[DS_STUD_ASSESSMENTS]
        df_ass_not_banked = df_ass.loc[df_ass['is_banked'] == 0]
        self.logger.debug("All students: %s Not banked: %s", str(len(df_ass)), str(len(df_ass_not_banked)))
        df = pd.merge(df_studinfo, df_ass_not_banked, how='left', on='id_student')
        df.fillna(-1, inplace=True)
        self.logger.debug("StudInfo Before merge: %s ... After merge: %s", str(len(df_studinfo)), str(len(df)))

        return df

    def __filter_submitted_until_date(self, date, df_students=None, dfs=None):
        self.logger.debug("Filtering students that submitted until the date: %s", date)
        if df_students is None:
            df_students = dfs[DS_STUD_INFO]

        df_stud_ass = dfs[DS_STUD_ASSESSMENTS]

        arr_submitted = df_stud_ass.loc[df_stud_ass['date_submitted'] <= date]['id_student']
        print("Removed:{}".format(len(arr_submitted)))
        df_filtered = df_students.loc[~df_students['id_student'].isin(arr_submitted)]
        return df_filtered

    def __get_submitted_and_remap(self, date, dfs=None):
        self.logger.debug("Getting already submitted students until the date: %s", date)
        df_stud_ass = dfs[DS_STUD_ASSESSMENTS]
        df_sa_submitted = df_stud_ass[df_stud_ass.date_submitted <= date]
        arr_submitted = df_sa_submitted['id_student']
        df_vle = dfs[DS_STUD_VLE]

        dfx = df_vle[df_vle.id_student.isin(arr_submitted)].reset_index().merge(df_sa_submitted, on='id_student')
        dfx['date_back'] = dfx.date_submitted - dfx.date
        dfx = dfx.set_index(['code_module', 'code_presentation'])
        dfx = dfx[['id_student', 'id_site', 'date', 'date_back', 'sum_click']]
        return dfx

    def __retrieve_registered_by_date(self, registration_date, df_students=None, dfs=None):
        """
        Retrieves only students that were registered in the given date - i.e. 1) they never unregister from the course
        or 2) they unregister after the specified date
        :param registration_date:
        :param df_students:
        :return:
        """
        self.logger.debug("Filtering by registration date -- not registered in the window will be filtered.")
        df_filtered = dfs[DS_STUD_REG].loc[(dfs[DS_STUD_REG]["date_unregistration"] > registration_date) | (
            dfs[DS_STUD_REG]["date_unregistration"].isnull())]
        self.logger.debug("DF_reg before join: %s", str(len(df_filtered)))
        if df_students is not None:
            self.logger.debug("DF_STUDENTS before join: %s", str(len(df_students)))
            df_filtered = pd.merge(df_students, df_filtered, on='id_student')
        self.logger.debug("DF_reg after join: %s", str(len(df_filtered)))
        return df_filtered

    def __load_config_and_datasets(self):
        self.logger.debug("Checking hdf5 file:%s", self.hdf5_path)
        self.check_hdf5(reload=False)
        self.__config = self._load_config_lazy(force_reload=True)
        self.logger.debug("Preparing data frames - filtering by mod-pres %s/%s", self.module, self.presentation)
        self.prepare_df_by_modpres()
        self.data_hdf5_manager.store_config(self.__config)

    def _load_config_lazy(self, force_reload=False):
        self.logger.debug("Trying to load config from HDF")
        if force_reload:
            self.logger.debug("Forcing reload of config")
            return self.__load_config()

        c = self.data_hdf5_manager.load_config()
        if c is not None:
            self.logger.debug("Config loaded from HDF5 features file.")
            return c
        else:
            self.logger.debug("Config not found in HDF, loading from raw data.")
            return self.__load_config()

    def __load_config(self, force_remote=False):
        self.logger.debug("Loading config for OULAD")
        self.logger.debug("Module: %s", self.module)
        self.logger.debug("Presentation: %s Train: %s", self.presentation, self.presentation_train)

        df_assessments = self.store_manager.load_dataframe(DS_ASSESSMENTS).sort_index()
        df_test = df_assessments.ix[self.module, self.presentation]
        df_train = df_assessments.ix[self.module, self.presentation_train]

        config = Config()

        if self.problem_definition.problem_definition_type == ProblemDefinition.ASSESSMENT:
            try:
                assessment_row_train = df_train.loc[df_train['assessment_name'] == self.assessment_name].iloc[0]
                assessment_row_test = df_test.loc[df_test['assessment_name'] == self.assessment_name].iloc[0]
                days_to_cutoff = self.problem_definition.days_to_cutoff
            except IndexError:
                raise Exception("The provided assesment name does not exist: ", self.assessment_name)
        else:
            days_to_cutoff = 1
            raise Exception("Not implemented")

        # id_assessment = assessment_row['id_assessment']
        assessment_name = assessment_row_train['assessment_name']
        days_to_predict = self.problem_definition.days_to_predict
        days_for_label_window = self.problem_definition.days_for_label_window

        config.cutoff_date_train = assessment_row_train['date']
        config.cutoff_date_test = assessment_row_test['date']
        config.pres_start = 1
        config.current_date_train = int(config.cutoff_date_train - days_to_cutoff)
        config.current_date_test = int(config.cutoff_date_test - days_to_cutoff)
        config.id_assessment_train = assessment_row_train['id_assessment']
        config.id_assessment_test = assessment_row_test['id_assessment']
        config.assessment_name = assessment_name
        config.test_labels_to = config.cutoff_date_test - days_to_cutoff + days_to_predict
        config.test_labels_from = config.test_labels_to - days_to_cutoff

        training_type = self.problem_definition.training_type
        if training_type == TrainingType.SELFLEARNER:
            config.train_labels_to = config.test_labels_from - 1
            config.train_labels_from = config.train_labels_to - days_for_label_window
        elif training_type == TrainingType.PREVIOUS_PRES:
            config.train_labels_to = config.test_labels_to
            config.train_labels_from = config.test_labels_from
        elif training_type == TrainingType.GOLDEN_STANDARD:
            config.train_labels_to = config.test_labels_to
            config.train_labels_from = config.test_labels_from

        self.logger.debug("Config loaded")
        logging.info("%s", vars(config))
        return config

    def extract_features(self, features):
        self.__load_config_and_datasets()
        self.logger.debug("Extracting  features")
        df_filtered_students_train = self.retrieve_filtered_students(self.dfs_train)
        df_filtered_students_test = self.retrieve_filtered_students(self.dfs_test)
        df_filtered_students_test = self.__retrieve_registered_by_date(self.__config.test_labels_from,
                                                                       df_students=df_filtered_students_test,
                                                                       dfs=self.dfs_test)

        print("Train: {}".format(str(len(df_filtered_students_train))))

        # Filter by assessment submission
        df_filtered_students_test = self.__filter_submitted_until_date(self.__config.test_labels_from - 1,
                                                                       df_filtered_students_test, dfs=self.dfs_test)

        df_filtered_students_train = self.__filter_submitted_until_date(self.__config.train_labels_from - 1,
                                                                        df_filtered_students_train, dfs=self.dfs_train)

        print("Train: {}".format(str(len(df_filtered_students_train))))
        print("Test: {}".format(str(len(df_filtered_students_test))))

        # 1. Labels
        df_train_labels = self.__retrieve_labels_submitted(df_filtered_students_train, self.__config.train_labels_to)
        df_test_labels = self.__retrieve_labels_submitted(df_filtered_students_test, self.__config.test_labels_to)

        self.data_hdf5_manager.store_df('train/y', df_train_labels)
        self.data_hdf5_manager.store_df('test/y', df_test_labels)

        # 2. demographic
        df_demog_train = self.__load_demog(df_train_labels[['id_student']], self.dfs_train)
        df_demog_test = self.__load_demog(df_test_labels[['id_student']], self.dfs_test)
        self.data_hdf5_manager.store_df('train/x_demog', df_demog_train)
        self.data_hdf5_manager.store_df('test/x_demog', df_demog_test)

        df_filtered_students_train = df_filtered_students_train[['id_student']]
        df_filtered_students_test = df_filtered_students_test[['id_student']]

        features_train_from_date = -50
        features_train_to_date = self.__config.train_labels_from - 1
        features_test_from_date = -50
        features_test_to_date = self.__config.test_labels_from - 1
        max_day = 50

        # df_vle = self.dfs[DS_VLE]
        df_stud_vle_train = self.dfs_train[DS_STUD_VLE].copy()
        df_stud_vle_test = self.dfs_test[DS_STUD_VLE].copy()

        df_stud_vle_train['date_back'] = features_train_to_date - df_stud_vle_train['date'] + 1
        df_stud_vle_test['date_back'] = features_test_to_date - df_stud_vle_test['date'] + 1

        if self.include_submitted:
            print("including submitted")
            print("First deadline:{} second:{}".format(self.__config.test_labels_from - 1, self.__config.train_labels_to))

            df_vle_submitted_train = self.__get_submitted_and_remap(self.__config.test_labels_from - 1,
                                                                    dfs=self.dfs_train)
            print("Len submitted: {} before: {}".format(len(df_vle_submitted_train), len(df_stud_vle_train)))
            df_stud_vle_train = df_stud_vle_train.append(df_vle_submitted_train)
            print("Len submitted: {} after: {}".format(len(df_vle_submitted_train), len(df_stud_vle_train)))
            print("counts before:{}".format(df_train_labels.groupby('submitted').size()))
            df_train_labels = self.__retrieve_labels_submitted(self.retrieve_filtered_students(self.dfs_train),
                                                               self.__config.train_labels_to)
            print("counts after:{}".format(df_train_labels.groupby('submitted').size()))

            self.data_hdf5_manager.store_df('train/y', df_train_labels)
            df_demog_train = self.__load_demog(df_train_labels[['id_student']], self.dfs_train)
            self.data_hdf5_manager.store_df('train/x_demog', df_demog_train)
            df_filtered_students_train = df_train_labels[['id_student']].drop_duplicates()

        print("Train2: {}".format(str(len(df_filtered_students_train))))

        df_stud_vle_date_filter_train = df_stud_vle_train.loc[
            (df_stud_vle_train.date >= features_train_from_date) & (
                df_stud_vle_train.date <= features_train_to_date)]
        df_stud_vle_date_filter_test = df_stud_vle_test.loc[
            (df_stud_vle_test.date >= features_test_from_date) & (
                df_stud_vle_test.date <= features_test_to_date)]
        # df_stud_vle_date_filter_train['date_back'] = features_train_to_date - df_stud_vle_date_filter_train['date'] + 1
        df_stud_vle_date_filter_train = df_stud_vle_date_filter_train.loc[
            df_stud_vle_date_filter_train['date_back'] <= max_day]
        # df_stud_vle_date_filter_test['date_back'] = features_test_to_date - df_stud_vle_date_filter_test['date'] + 1
        df_stud_vle_date_filter_test = df_stud_vle_date_filter_test.loc[
            df_stud_vle_date_filter_test['date_back'] <= max_day]




        # 3. VLE - totals sums
        df_day_train = self.__extract_day_sums(df_stud_vle_date_filter_train, df_filtered_students_train)
        df_day_test = self.__extract_day_sums(df_stud_vle_date_filter_test, df_filtered_students_test)
        self.data_hdf5_manager.store_df('train/x_vle_day', df_day_train)
        self.data_hdf5_manager.store_df('test/x_vle_day', df_day_test)

        # 4. VLE - totals flags
        df_day_flags_train = self.__extract_day_flags(df_stud_vle_date_filter_train, df_filtered_students_train,
                                                      df_day_sums=df_day_train)
        df_day_flags_test = self.__extract_day_flags(df_stud_vle_date_filter_test, df_filtered_students_test,
                                                     df_day_sums=df_day_test)
        self.data_hdf5_manager.store_df('train/x_vle_day_flags', df_day_flags_train)
        self.data_hdf5_manager.store_df('test/x_vle_day_flags', df_day_flags_test)

        # 5. VLE - site_type sums
        df_day_activity_type_train = self.__extract_day_activitytype(df_stud_vle_date_filter_train,
                                                                     df_filtered_students_train,
                                                                     self.dfs_train)
        df_day_activity_type_test = self.__extract_day_activitytype(df_stud_vle_date_filter_test,
                                                                    df_filtered_students_test,
                                                                    self.dfs_test)
        self.data_hdf5_manager.store_df('train/x_vle_day_activity_type', df_day_activity_type_train)
        self.data_hdf5_manager.store_df('test/x_vle_day_activity_type', df_day_activity_type_test)

        # 6. VLE - site/activity_type flag
        df_day_activity_type_flags_train = self.__extract_day_activitytype_flags(df_stud_vle_date_filter_train,
                                                                                 df_filtered_students_train,
                                                                                 df_day_activity_type_sums=df_day_activity_type_train)
        df_day_activity_type_flags_test = self.__extract_day_activitytype_flags(df_stud_vle_date_filter_test,
                                                                                df_filtered_students_test,
                                                                                df_day_activity_type_sums=df_day_activity_type_test)
        self.data_hdf5_manager.store_df('train/x_vle_day_activity_type_flags', df_day_activity_type_flags_train)
        self.data_hdf5_manager.store_df('test/x_vle_day_activity_type_flags', df_day_activity_type_flags_test)

        # 7. VLE - various
        # Some transformations needed, non trivial
        df_vle_statistics_train = self.__extract_vle_statistics(df_stud_vle_date_filter_train,
                                                                df_filtered_students_train)
        df_vle_statistics_test = self.__extract_vle_statistics(df_stud_vle_date_filter_test, df_filtered_students_test)
        self.data_hdf5_manager.store_df('train/x_vle_statistics', df_vle_statistics_train)
        self.data_hdf5_manager.store_df('test/x_vle_statistics', df_vle_statistics_test)

        # VLE statistict before presentation start
        df_vle_statistics_before_train = self.__extract_vle_statistics_before_start(df_stud_vle_date_filter_train,
                                                                                    df_filtered_students_train)
        df_vle_statistics_before_test = self.__extract_vle_statistics_before_start(df_stud_vle_date_filter_test,
                                                                                   df_filtered_students_test)
        self.data_hdf5_manager.store_df('train/x_vle_statistics_beforestart', df_vle_statistics_before_train)
        self.data_hdf5_manager.store_df('test/x_vle_statistics_beforestart', df_vle_statistics_before_test)

        # 8. VLE - study plan
        # Is this necessary?

        # 9. Registration features
        df_reg_stats_train = self.__extract_reg_stats(df_filtered_students_train, self.dfs_train)
        df_reg_stats_test = self.__extract_reg_stats(df_filtered_students_test, self.dfs_test)
        self.data_hdf5_manager.store_df('train/x_reg_statistics', df_reg_stats_train)
        self.data_hdf5_manager.store_df('test/x_reg_statistics', df_reg_stats_test)

        # Joining features
        self.__join_features(features)

        return self.data

    def __join_features(self, features):
        # Y - labels
        self.data["y_train"] = self.data_hdf5_manager.load_df('train/y')
        self.data["y_test"] = self.data_hdf5_manager.load_df('test/y')
        self.data["x_train"] = self.data["y_train"][['id_student']]
        self.data["x_test"] = self.data["y_test"][['id_student']]

        if len(features) < 1:
            raise Warning("No features given")

        for feature in features:
            feature = '_'.join(['x', feature])
            self.data["x_train"] = pd.merge(self.data["x_train"],
                                            self.data_hdf5_manager.load_df('/'.join(['train', feature])),
                                            on='id_student')
            self.logger.debug("x train size: %s columns: %s", len(self.data["x_train"]),
                              len(self.data["x_train"].columns))
            self.data["x_test"] = pd.merge(self.data["x_test"],
                                           self.data_hdf5_manager.load_df('/'.join(['test', feature])),
                                           on='id_student')
            self.logger.debug("x test size: %s columns: %s", len(self.data["x_test"]), len(self.data["x_test"].columns))

        # X + Y together
        self.data["all_train"] = pd.merge(self.data["x_train"], self.data["y_train"], on='id_student')
        self.data["all_test"] = pd.merge(self.data["x_test"], self.data["y_test"], on='id_student')

    def __extracty_weekly_features(self, df_students):
        return df_students

    def __extract_day_sums(self, df_stud_vle_date_filter, ds_students):
        df_date_grouped = df_stud_vle_date_filter.groupby(['id_student', 'date_back'])[['sum_click']].sum()
        df_date_grouped = df_date_grouped.unstack()
        df_date_grouped_merged = pd.merge(ds_students, df_date_grouped, how='left', left_on='id_student',
                                          right_index=True)
        df_date_grouped_merged.fillna(0, inplace=True)
        df_date_grouped_merged.set_index('id_student', inplace=True)
        df_date_grouped_merged.rename(columns=lambda x: '_'.join([str(a) for a in x]), inplace=True)
        df_date_grouped_merged.reset_index(inplace=True)
        logging.info("COLUMN COUNT:VLE DAY SUM: {}".format(len(df_date_grouped_merged.columns)))
        return df_date_grouped_merged

    def __extract_day_activitytype(self, df_stud_vle_date_filter, ds_students, dfs):
        df_activity_type = pd.merge(df_stud_vle_date_filter, dfs[DS_VLE], on='id_site')
        df_day_activity_type_grouped = df_activity_type.groupby(['id_student', 'activity_type', 'date_back'])[
            ['sum_click']].sum()
        df_day_activity_type_grouped = df_day_activity_type_grouped.unstack().unstack()
        df_day_activity_type_grouped_merged = pd.merge(ds_students, df_day_activity_type_grouped, how='left',
                                                       left_on='id_student',
                                                       right_index=True)
        df_day_activity_type_grouped_merged.set_index('id_student', inplace=True)
        df_day_activity_type_grouped_merged.rename(columns=lambda x: '_'.join([str(a) for a in x]), inplace=True)
        df_day_activity_type_grouped_merged.reset_index(inplace=True)
        df_day_activity_type_grouped_merged.fillna(0, inplace=True)
        logging.info("COLUMN COUNT:VLE DAY ACTIVITY TYPE SUM: %s", len(df_day_activity_type_grouped_merged.columns))
        return df_day_activity_type_grouped_merged

    def __retrieve_labels_submitted(self, df_students, cutoff):
        # Implementing the interface
        self.logger.debug("Cutoff: %s", cutoff)
        label_submitted = 'submitted'
        label_submit_in = 'submit_in'
        df_students.loc[:, label_submitted] = df_students.apply(
            lambda row: 1 if row['date_submitted'] <= cutoff else 0, axis=1)
        df_students.loc[:, label_submit_in] = df_students.apply(
            lambda row: cutoff - row['date_submitted'] if row[label_submitted] == 1 else -1, axis=1)
        df_labels = df_students[['id_student', label_submitted, label_submit_in]]
        return df_labels

    def __load_demog(self, df_students, dfs):
        self.logger.debug("Loading demographic features")
        demog_attrs = ['id_student', 'gender', 'region', 'highest_education', 'imd_value', 'age_band',
                       'num_of_prev_attempts', 'studied_credits',
                       'disability']

        df_demog = pd.merge(dfs[DS_STUD_INFO], df_students, on='id_student')
        df_demog = FeaturesMapping().map_imd_band(df_demog)
        df_demog = df_demog[demog_attrs]
        logging.info("COLUMN COUNT:DEMOGRAPHIC: {}".format(len(df_demog.columns)))
        return df_demog

    def __extract_day_activitytype_flags(self, df_stud_vle_date_filter, df_filtered_students,
                                         df_day_activity_type_sums=None) -> pd.DataFrame:
        """
            Extracts the daily activity type flags from the given dataframe.
        """
        if df_day_activity_type_sums is None:
            df_day_activity_type_sums = self.__extract_day_activitytype(df_stud_vle_date_filter)
        df_date_grouped_merged_flags = df_day_activity_type_sums.copy()
        df_date_grouped_merged_flags.set_index(['id_student'], inplace=True)
        df_date_grouped_merged_flags.rename(
            columns=lambda x: re.sub('sum_click', 'is_click', '_'.join([str(a) for a in x]) if type(x) is tuple else x),
            inplace=True)
        df_date_grouped_merged_flags[df_date_grouped_merged_flags > 0] = 1
        df_date_grouped_merged_flags.reset_index(inplace=True)
        logging.info("COLUMN COUNT:VLE DAY ACTIVITY TYPE FLAG: {}".format(len(df_date_grouped_merged_flags.columns)))
        return df_date_grouped_merged_flags

    def __extract_day_flags(self, df_stud_vle_date_filter, df_filtered_students, df_day_sums=None):
        if df_day_sums is None:
            df_day_sums = self.__extract_day_sums(df_stud_vle_date_filter, df_filtered_students)
        df_date_grouped_merged_flags = df_day_sums.copy()
        df_date_grouped_merged_flags.set_index(['id_student'], inplace=True)
        df_date_grouped_merged_flags.rename(
            columns=lambda x: re.sub('sum_click', 'is_click', '_'.join([str(a) for a in x]) if type(x) is tuple else x),
            inplace=True)
        df_date_grouped_merged_flags[df_date_grouped_merged_flags > 0] = 1
        df_date_grouped_merged_flags.reset_index(inplace=True)
        logging.info("COLUMN COUNT:VLE DAY FLAGS: {}".format(len(df_date_grouped_merged_flags.columns)))
        return df_date_grouped_merged_flags

    def __extract_vle_statistics_before_start(self, df_stud_vle_date_filter, ds_students):
        # max_click_day = df_stud_vle_date_filter['date'].max()
        min_click_day = df_stud_vle_date_filter['date'].min()
        presentation_start_day = 0
        num_days_before_start = math.fabs(min_click_day - presentation_start_day)

        # before start of the presentation
        df_stud_vle_date_filter_before_start = df_stud_vle_date_filter.loc[
            df_stud_vle_date_filter.date < presentation_start_day]
        df_date_sums = df_stud_vle_date_filter_before_start.groupby(['id_student', 'date'])[['sum_click']].agg(
            ['count', 'sum']).reset_index()
        df_date_sums.columns = ['id_student', 'date', 'count_materials', 'sum_click']
        df_date_visit_stats_beforestart_active = df_date_sums.groupby(['id_student']).agg(
            {'count_materials': {'count_days': 'count', 'sum_material': 'sum', 'max_materials': 'max',
                                 'min_materials:': 'min',
                                 'median_materials': 'median', 'avg_materials': 'mean'},
             'sum_click': {'sum_click': 'sum', 'max_clicks': 'max', 'min_clicks': 'min', 'median_clicks': 'median',
                           'avg_clicks': 'mean'}})
        df_date_visit_stats_beforestart_active.columns = ['count_days_beforestart', 'sum_material_beforestart',
                                                          'max_material_beforestart',
                                                          'avg_material_count_beforestart_peractive',
                                                          'median_material_count_beforestart_peractive',
                                                          'min_material_count_beforestart_peractive',
                                                          'median_clicks_beforestart_peractive',
                                                          'avg_clicks_beforestart_peractive',
                                                          'min_clicks_beforestart_peractive', 'sum_click_beforestart',
                                                          'max_clicks_beforestart'
                                                          ]
        df_date_visit_stats_beforestart_active['count_days_beforestart_rel'] = df_date_visit_stats_beforestart_active[
                                                                                   'count_days_beforestart'] / num_days_before_start

        df_date_visit_stats_beforestart_active = pd.merge(ds_students, df_date_visit_stats_beforestart_active,
                                                          how='left',
                                                          left_on='id_student',
                                                          right_index=True)
        df_date_visit_stats_beforestart_active.fillna(0, inplace=True)

        # all days
        df_date_sums.set_index(['id_student', 'date'], inplace=True)
        df_date_sums = df_date_sums.unstack().fillna(0).stack()
        df_date_sums.reset_index(inplace=True)
        df_date_visit_stats_beforestart = df_date_sums.groupby(['id_student']).agg(
            {'count_materials': {'min_materials:': 'min',
                                 'median_materials': 'median', 'avg_materials': 'mean'},
             'sum_click': {'min_clicks': 'min', 'median_clicks': 'median', 'avg_clicks': 'mean'}})
        df_date_visit_stats_beforestart.columns = [
            'avg_material_count_beforestart',
            'median_material_count_beforestart', 'min_material_count_beforestart',
            'median_clicks_beforestart', 'avg_clicks_beforestart',
            'min_clicks_beforestart'
        ]

        df_date_visit_stats_fromstart = pd.merge(ds_students, df_date_visit_stats_beforestart, how='left',
                                                 left_on='id_student',
                                                 right_index=True)
        df_date_visit_stats_fromstart.fillna(0, inplace=True)

        # join
        ret_val = pd.merge(df_date_visit_stats_beforestart_active, df_date_visit_stats_fromstart,
                           left_on='id_student', right_on='id_student')
        logging.info("COLUMN COUNT:VLE STATISTICS BEFORE_START: {}".format(len(ret_val.columns)))

        return ret_val

    def __extract_vle_statistics(self, df_stud_vle_date_filter, ds_students, num_days_from_vleopen=None):
        max_click_day = df_stud_vle_date_filter['date'].max()
        min_click_day = df_stud_vle_date_filter['date'].min()
        presentation_start_day = 0

        if num_days_from_vleopen is None:
            num_days_from_vleopen = max_click_day - min_click_day + 1

        num_days_from_start = max_click_day - presentation_start_day + 1
        self.logger.debug("Num days: %s", num_days_from_vleopen)

        df_last_login = df_stud_vle_date_filter.groupby(['id_student']).agg(
            {'date': {'first_login': 'min', 'last_login': 'max'},
             'date_back': {'last_login_rel': 'min'}})
        df_last_login.columns = df_last_login.columns.droplevel(0)
        df_last_login = pd.merge(ds_students, df_last_login, how='left', left_on='id_student',
                                 right_index=True)
        # df_last_login[['first_login']] = df_last_login[['first_login']].fillna(1000)
        # df_last_login[['last_login']] = df_last_login[['last_login']].fillna(-1000)
        # df_last_login[['last_login_rel']] = df_last_login[['last_login_rel']].fillna(1000)
        df_last_login['never_logged'] = df_last_login['first_login'].isnull() * 1.0

        # From VLE open
        df_date_sums = df_stud_vle_date_filter.groupby(['id_student', 'date'])[['sum_click']].agg(
            ['count', 'sum']).reset_index()
        df_date_sums.columns = ['id_student', 'date', 'count_materials', 'sum_click']
        df_date_visit_stats_fromvleopen = df_date_sums.groupby(['id_student']).agg(
            {'count_materials': {'count_days': 'count', 'sum_material': 'sum'}, 'sum_click': {'sum_click': 'sum'}})
        df_date_visit_stats_fromvleopen.columns = ['count_days_fromvleopen', 'sum_material_fromvleopen',
                                                   'sum_click_fromvleopen']
        df_date_visit_stats_fromvleopen['count_days_fromvleopen_rel'] = df_date_visit_stats_fromvleopen[
                                                                            'count_days_fromvleopen'] / num_days_from_vleopen
        df_date_visit_stats_fromvleopen = pd.merge(ds_students, df_date_visit_stats_fromvleopen, how='left',
                                                   left_on='id_student',
                                                   right_index=True)
        df_date_visit_stats_fromvleopen.fillna(0, inplace=True)

        ret_val = pd.merge(df_date_visit_stats_fromvleopen, df_last_login, on='id_student')

        if max_click_day >= presentation_start_day:
            df_stud_vle_date_filter2 = df_stud_vle_date_filter.loc[
                df_stud_vle_date_filter.date >= presentation_start_day]
            df_date_sums = df_stud_vle_date_filter2.groupby(['id_student', 'date'])[['sum_click']].agg(
                ['count', 'sum']).reset_index()
            df_date_sums.columns = ['id_student', 'date', 'count_materials', 'sum_click']
            df_date_visit_stats_fromstart_active = df_date_sums.groupby(['id_student']).agg(
                {'count_materials': {'count_days': 'count', 'sum_material': 'sum', 'max_materials': 'max',
                                     'min_materials:': 'min',
                                     'median_materials': 'median', 'avg_materials': 'mean'},
                 'sum_click': {'sum_click': 'sum', 'max_clicks': 'max', 'min_clicks': 'min', 'median_clicks': 'median',
                               'avg_clicks': 'mean'}})
            df_date_visit_stats_fromstart_active.columns = ['count_days_fromstart', 'sum_material_fromstart',
                                                            'max_material_fromstart',
                                                            'avg_material_count_fromstart_peractive',
                                                            'median_material_count_fromstart_peractive',
                                                            'min_material_count_fromstart_peractive',
                                                            'median_clicks_fromstart_peractive',
                                                            'avg_clicks_fromstart_peractive',
                                                            'min_clicks_fromstart_peractive', 'sum_click_fromstart',
                                                            'max_clicks_fromstart'
                                                            ]
            df_date_visit_stats_fromstart_active['count_days_fromstart_rel'] = df_date_visit_stats_fromstart_active[
                                                                                   'count_days_fromstart'] / num_days_from_start

            df_date_visit_stats_fromstart_active = pd.merge(ds_students, df_date_visit_stats_fromstart_active,
                                                            how='left',
                                                            left_on='id_student',
                                                            right_index=True)
            df_date_visit_stats_fromstart_active.fillna(0, inplace=True)

            # all days
            df_date_sums.set_index(['id_student', 'date'], inplace=True)
            df_date_sums = df_date_sums.unstack().fillna(0).stack()
            df_date_sums.reset_index(inplace=True)
            df_date_visit_stats_fromstart = df_date_sums.groupby(['id_student']).agg(
                {'count_materials': {'min_materials:': 'min',
                                     'median_materials': 'median', 'avg_materials': 'mean'},
                 'sum_click': {'min_clicks': 'min', 'median_clicks': 'median', 'avg_clicks': 'mean'}})
            df_date_visit_stats_fromstart.columns = [
                'avg_material_count_fromstart',
                'median_material_count_fromstart', 'min_material_count_fromstart',
                'median_clicks_fromstart', 'avg_clicks_fromstart',
                'min_clicks_fromstart'
            ]

            df_date_visit_stats_fromstart = pd.merge(ds_students, df_date_visit_stats_fromstart, how='left',
                                                     left_on='id_student',
                                                     right_index=True)
            df_date_visit_stats_fromstart.fillna(0, inplace=True)

            # join
            ret_val = pd.merge(ret_val, df_date_visit_stats_fromstart_active, left_on='id_student',
                               right_on='id_student')
            ret_val = pd.merge(ret_val, df_date_visit_stats_fromstart, left_on='id_student', right_on='id_student')

        # Consecutive days
        df_stud_date = \
            df_stud_vle_date_filter.groupby(['id_student', 'date_back'])[['sum_click']].count().reset_index()[
                ['id_student', 'date_back']]
        df_consecutive_days = df_stud_date.groupby(['id_student']).agg(
            {'date_back': {'consecutive_days': FeaturesMapping.consecut}})
        df_consecutive_days.columns = ['consecutive_days']
        ret_val = pd.merge(ret_val, df_consecutive_days, left_on='id_student', how='left', right_index=True)
        ret_val.fillna(0, inplace=0)

        self.logger.debug("1st: %s, 2nd: %s, 3rd: %s", len(df_last_login), len(df_date_visit_stats_fromvleopen),
                          len(ret_val))
        logging.info("COLUMN COUNT:VLE STATISTICS: %s", len(ret_val.columns))
        return ret_val

    def __extract_reg_stats(self, ds_students, dfs):
        df_reg = dfs[DS_STUD_REG].copy().reset_index()
        df_reg_stats = df_reg[['id_student', 'date_registration']]
        df_reg_stats = pd.merge(ds_students, df_reg_stats, on='id_student')
        logging.info("COLUMN COUNT:REGISTRATION STATISTICS: %s", len(df_reg_stats.columns))
        return df_reg_stats


# MAIN script

# from pandas.io.pytables import HDFStore
# hdf5_path = 'data_load/data/oulad.h5'
# module = 'AAA'
# pres = '2013J'
# id_ass = 1752
# store = HDFStore(hdf5_path)
# df_ass = store['assessments'].ix[module, pres]
# df_stud_reg = store['studentRegistration'].ix[module, pres]
# df_stud_ass = store['studentAssessment'].ix[id_ass]
# df_stud_vle = store['studentVle'].ix[module, pres]
# df_vle = store['vle'].ix[module, pres]
# df_studinfo = store['studentInfo'].ix[module, pres]
# store.close()


def main():
    print(os.path.dirname(__file__))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    problemDef = ProblemDefinition("FFF", "2014J", "TMA 1", 0, 5, 0)
    FeatureExtractionOulad(problemDef).extract_features(features=["vle_statistics", "demog"])


if __name__ == "__main__":
    main()
