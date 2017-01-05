import logging
__author__ = 'author'


class Config:
    # config_directory = "config"

    cutoff_date_train = 0
    cutoff_date_test = 0
    pres_start = 0
    id_assessment_train = 0
    id_assessment_test = 0
    assessment_name = ""
    days_to_cutoff = 0
    current_date_train = 0
    current_date_test = 0
    train_labels_from = 0
    train_labels_to = 0
    test_labels_from = 0
    test_labels_to = 0

    def __init__(self):
        """
        """

    @staticmethod
    def from_pytable_row(row):
        c = Config()
        logging.debug("Creating config from row:")
        attr_names = [att for att in dir(c) if not att.startswith('__') and not callable(getattr(c, att))]
        for att in attr_names:
            logging.debug("Storing att: %s value:%s", att, row[att])
            setattr(c, att, row[att])
        return c


class ConfigLoader:

    def __init__(self, problem_definition):
        self.config = Config(problem_definition)

    def load_config(self, force_remote=False):
        raise NotImplemented
