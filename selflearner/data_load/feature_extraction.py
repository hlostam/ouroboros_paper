# import logging
import os

# import pandas as pd

# from selflearner.data_load.config_loader import Config


class FeatureExtraction:
    def __init__(self, problem_definition):
        self.problem_definition = problem_definition
        self.module = problem_definition.module
        self.presentation = problem_definition.presentation
        self.presentation_train = problem_definition.presentation_train
        self.assessment_name = problem_definition.assessment_name
        self.feature_directory = os.path.join(os.sep, os.path.dirname(__file__), self.data_directory,
                                              problem_definition.string_representation)
        self.days_to_cutoff = problem_definition.days_to_cutoff
        self.offset_days = problem_definition.offset_days
        self.features_days = problem_definition.features_days
        self._init_data()
        self.is_data_initialised = False

    def _init_data(self):
        self.data = dict()
        self.data["train"] = {}
        self.data["test"] = {}
        self.data["train"]["x"] = {}
        self.data["train"]["_all"] = {}
        self.data["train"]["y"] = {}
        self.data["test"]["x"] = {}
        self.data["test"]["_all"] = {}
        self.data["test"]["y"] = {}
        self.data["train_all"] = {}
        self.data["test_all"] = {}

    # abstract methods
    def prepare_student_data(self, source_type):
        raise NotImplementedError("method not implemented.")

    # Common methods

    # def extract_features_inner(self, source_type, features):
    #     # self.prepare_student_data(source_type)
    #
    #     self.extract_y(source_type)
    #
    #     for feature in features:
    #         self.get_feature(source_type, feature)

    # def join_features(self, source_type="train"):
    #     """Joins all the features in the dataset into _all data"""
    #     logging.debug("Joining features: %s", source_type)
    #     logging.debug("%s", self.data.keys())
    #     self.data[source_type]["_all"] = self.data[source_type]["y"]
    #     for d in self.data[source_type]["x"].keys():
    #         if d == "_all":
    #             continue
    #         logging.debug("Joining feature: %s", d)
    #         self.data[source_type]["_all"] \
    #             = pd.merge(self.data[source_type]["_all"], self.data[source_type]["x"][d], on="id_student")

    # def get_feature(self, source_type, feature):
    #     feature_path = self.get_dataset_path('_'.join(['x', feature]), source_type)
    #
    #     if feature == "demog":
    #         self.get_demog(feature_path, source_type)
    #     elif feature == "vle_sums":
    #         self.get_vle_sums(feature_path, source_type)
    #     elif feature == "vle_flags":
    #         self.get_vle_flags(feature_path, source_type)
    #     elif feature == "vle_sitetype_sums":
    #         self.get_vle_sitetype_sums(feature_path, source_type)
    #     elif feature == "vle_sitetype_flags":
    #         self.get_vle_sitetype_flags(feature_path, source_type)
    #     elif feature == "vle_consec_days":
    #         self.get_vle_consec_days(feature_path, source_type)
    #     elif feature == "vle_study_plan":
    #         self.get_vle_study_plan(feature_path, source_type)
    #     else:
    #         return
    #
    #     self.data[source_type]["x"][feature] = pd.read_csv(feature_path)

    # def join_features(self, source_type="train"):
    #     """Joins all the features in the dataset into _all data"""
    #     logging.debug("Joining features: %s", source_type)
    #     logging.debug("%s", self.data.keys())
    #     self.data[source_type]["_all"] = self.data[source_type]["y"]
    #     for d in self.data[source_type]["x"].keys():
    #         if d == "_all":
    #             continue
    #         logging.debug("Joining feature: %s", d)
    #         self.data[source_type]["_all"] \
    #             = pd.merge(self.data[source_type]["_all"], self.data[source_type]["x"][d], on="id_student")

    def extract_features(self, features):
        raise NotImplementedError("method not implemented.")
