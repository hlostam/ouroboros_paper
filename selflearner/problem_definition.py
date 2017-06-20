__author__ = 'author'


class TrainingType:
    SELFLEARNER = 'selflearner'
    PREVIOUS_PRES = 'previous_pres'
    GOLDEN_STANDARD = 'golden_standard'


class ProblemDefinition:
    ASSESSMENT = 'assessment'
    MODPRES_DAY = 'days'

    def __init__(self, module, presentation, assessment_name, days_to_cutoff,
                 days_to_predict=None,
                 days_for_label_window=None,
                 features_days=0, day_of_presentation=None, y_column='submitted', grouping_column='submit_in',
                 id_column='id_student',
                 presentation_train=None, training_type=TrainingType.SELFLEARNER,
                 filter_only_registered=True,
                 submitted_append_min_date=0,
                 submitted_append_min_date_rel=100):
        """

        :param module: code of module (e.g. AAA)
        :param presentation: code of presentation (e.g. 2015J)
        :param assessment_name:
        :param days_to_cutoff:
        :param days_to_predict:
        :param days_for_label_window: the number of days that are used for measuring whether the student submitted the
               assignment or not, by default set to None, which means it will use days_to_predict.
        :param features_days:
        :param day_of_presentation:
        :param y_column:
        :param grouping_column:
        :param id_column:
        :param presentation_train:
        :param training_type:
        """
        self.module = module
        self.presentation = presentation
        self.assessment_name = assessment_name
        self.days_to_cutoff = days_to_cutoff
        if days_to_predict is None:
            days_to_predict = days_to_cutoff + 1
        self.days_to_predict = days_to_predict
        if days_for_label_window is None:
            days_for_label_window = days_to_predict
        self.days_for_label_window = days_for_label_window
        self.features_days = features_days
        self.y_column = y_column
        self.grouping_column = grouping_column
        self.id_column = id_column
        self.training_type = training_type
        self.presentation_train = presentation_train
        self.filter_only_registered = filter_only_registered
        self.submitted_append_min_date = submitted_append_min_date
        self.submitted_append_min_date_rel = submitted_append_min_date_rel

        if self.presentation_train is None:
            self.training_type = TrainingType.SELFLEARNER
            self.presentation_train = presentation

        if day_of_presentation:
            self.day_of_presentation = day_of_presentation
            if self.day_of_presentation == 0:
                self.day_of_presentation = 1
            self.string_representation = self.get_modpres_day_str(self.day_of_presentation)
            self.problem_definition_type = self.MODPRES_DAY
        else:
            self.string_representation = self.get_assessment_type_str(self.assessment_name, self.days_to_cutoff
                                                                      )
            self.problem_definition_type = self.ASSESSMENT

    def get_assessment_type_str(self, assessment_name, days_to_cutoff):
        return "/".join([self.module, self.presentation, self.presentation_train,
                         self.ASSESSMENT,
                         assessment_name, str(days_to_cutoff)]).replace(" ", "_")

    def get_modpres_day_str(self, day_of_presentation):
        return "/".join([self.module, self.presentation, self.presentation_train,
                         self.MODPRES_DAY,
                         day_of_presentation]).replace(" ", "_")
