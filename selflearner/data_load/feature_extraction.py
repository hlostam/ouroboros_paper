# import logging
import os

# import pandas as pd

# from selflearner.data_load.config_loader import Config
from selflearner.problem_definition import ProblemDefinition


class FeatureExtraction:
    def __init__(self, problem_definition: ProblemDefinition, include_submitted=False, submitted_append_min_date=-10,
                 submitted_append_min_date_rel=None):
        self.problem_definition = problem_definition
        self.module = problem_definition.module
        self.presentation = problem_definition.presentation
        self.presentation_train = problem_definition.presentation_train
        self.assessment_name = problem_definition.assessment_name
        self.feature_directory = os.path.join(os.sep, os.path.dirname(__file__), self.data_directory,
                                              problem_definition.string_representation)
        self.days_to_cutoff = problem_definition.days_to_cutoff
        self.days_to_predict = problem_definition.days_to_predict

        self.features_days = problem_definition.features_days
        self._init_data()
        self.is_data_initialised = False
        self.include_submitted = include_submitted
        self.submitted_append_min_date = submitted_append_min_date
        self.submitted_append_min_date_rel = submitted_append_min_date_rel

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

    def extract_features(self, features):
        raise NotImplementedError("method not implemented.")
