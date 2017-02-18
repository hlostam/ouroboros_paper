import logging

import pandas as pd
import numpy as np
import warnings

from selflearner.data_load.features_extraction_oulad import FeatureExtractionOulad
from selflearner.learning.learner import Learner
from selflearner.plotting.plotting import plot_df
from selflearner.problem_definition import ProblemDefinition, TrainingType
import seaborn as sns

# TODO: Unified settings somewhere from global
pd.set_option('display.max_columns', 0)  # Display any number of columns
pd.set_option('display.max_rows', 0)  # Display any number of rows
pd.set_option('expand_frame_repr', False)


class ClassifierResult:
    def __init__(self, metrics=None,
                 metrics_k=None):
        if metrics is None:
            metrics = ['auc', 'prec', 'recall', 'fscore', 'pr_auc']
        if metrics_k is None:
            metrics_k = ['top_k_prec', 'top_k_rec']
        self.metrics = {}
        self.metrics_k = {}
        for metric_k in metrics_k:
            self.metrics_k[metric_k] = {}
        self.features = []


class ExperimentResult:
    def __init__(self, problem_definition: ProblemDefinition):
        self.problem_definition = problem_definition
        self.class_numbers_train = {}
        self.class_numbers_test = {}
        self.classifier_results = {}
        self.feature_names = {}


# TODO: Fix the disambiguition with submitted/not_submitted label!!!!
def map_k_toclassname(k):
    if int(k) == 1:
        return 'S'
    else:
        return 'NS'


class MultiDayExperiment:
    def __init__(self, max_days=11, min_days=0, max_days_to_predict=None, count_all_days_to_predict=False,
                 days_for_label_window=None, count_all_days_for_label_window=False,
                 include_submitted=False, submitted_append_min_date=0,
                 features=None, problem_defintions: [ProblemDefinition] = None,
                 module_presentations=None, assessment_name=None,
                 grouping_column='submit_in', id_column='id_student',
                 training_type=TrainingType.SELFLEARNER,
                 label_name='submitted', sampler=None, classifiers=None, feature_extractors=None,
                 top_k_prec_list=None, optimise_threshold=True,
                 metrics=None, metrics_k=None,
                 filter_only_registered=True):
        if features is None:
            features = ["demog"]
        if feature_extractors is None:
            feature_extractors = []
        if max_days_to_predict is None:
            max_days_to_predict = max_days
        if top_k_prec_list is None:
            top_k_prec_list = [5, 10, 25, 50, 75]
        if metrics is None:
            metrics = ['auc', 'prec', 'recall', 'fscore', 'pr_auc']
        if metrics_k is None:
            metrics_k = ['top_k_prec', 'top_k_rec']
        if problem_defintions is not None:
            self.problem_definitions = problem_defintions
        else:
            self.problem_definitions = self.create_problem_definitions(module_presentations, assessment_name, min_days,
                                                                       max_days,
                                                                       label_name, grouping_column, id_column,
                                                                       training_type, max_days_to_predict,
                                                                       count_all_days_to_predict=count_all_days_to_predict,
                                                                       days_for_label_window=days_for_label_window,
                                                                       count_all_days_for_label_window=count_all_days_for_label_window,
                                                                       filter_only_registered=filter_only_registered)
        self.max_days = max_days
        self.max_days_to_predict = max_days_to_predict
        self.min_days = min_days
        self.count_all_days_to_predict = count_all_days_to_predict
        self.days_for_label_window = days_for_label_window
        self.count_all_days_for_label_window = count_all_days_for_label_window
        self.features = features
        self.label_name = label_name
        self.sampler = sampler
        self.topkpreclist = top_k_prec_list
        self.optimise_threshold = optimise_threshold
        self.metrics = metrics
        self.metrics_k = metrics_k
        self.learner_names = []
        self.learners = []
        self.feature_extractors = feature_extractors
        self.classifiers = classifiers
        if self.classifiers is not None:
            self.classifiers_names = [name for cl, name in self.classifiers]
        self.experiment_results = []
        self.include_submitted = include_submitted
        self.submitted_append_min_date = submitted_append_min_date
        self.filter_only_registered=filter_only_registered

    def metric_to_df(self, metric):
        df = pd.DataFrame(self.metrics[metric])
        df.columns = self.learner_names
        df.index_name = 'Day'
        return df

    def metric_k_to_df(self, metric_k, k):
        df = pd.DataFrame(self.metrics_k[metric_k][k])
        df.columns = self.learner_names
        df.index_name = 'Day'
        return df

    def perform_experiment(self):
        """
        Performs the whole experiment, iterates over days and list of problem definitions.
        """
        for problem_def in self.problem_definitions:
            logging.info(
                "{} {}, Day: {} Predicted day: {} LabelWindow: {}".format(problem_def.module, problem_def.presentation,
                                                                          problem_def.days_to_cutoff,
                                                                          problem_def.days_to_predict,
                                                                          problem_def.days_for_label_window))
            print("{} {}, Day: {} Predicted day: {} LabelWindow: {}".format(problem_def.module, problem_def.presentation,
                                                                            problem_def.days_to_cutoff,
                                                                            problem_def.days_to_predict,
                                                                            problem_def.days_for_label_window))
            problem_def = problem_def
            fe = FeatureExtractionOulad(problem_def,
                                        include_submitted=self.include_submitted,
                                        submitted_append_min_date=self.submitted_append_min_date)

            data = fe.extract_features(features=self.features)
            train_data = data["all_train"]
            test_data = data["all_test"]

            # Class counts
            experiment_result = ExperimentResult(problem_def)
            experiment_result.class_numbers_train = self._get_class_counts(train_data)
            experiment_result.class_numbers_test = self._get_class_counts(test_data)

            # Learningq

            if self.classifiers is not None:
                learner = Learner(train_data, test_data, self.label_name, problem_def, sampler=self.sampler,
                                  classifiers=self.classifiers, feature_extractors=self.feature_extractors,
                                  topk_prec_list=self.topkpreclist, optimise_threshold=self.optimise_threshold)
                learner.run()
                self.learners.append(learner)
                self.learner_names = learner.names
                learner_results = self.get_classifier_results(learner)
                experiment_result.feature_names = learner.col_names
                experiment_result.classifier_results = learner_results

            self.experiment_results.append(experiment_result)

    def add_metrics(self, learner: Learner):
        # add the evaluation metrics
        for key, value in self.metrics.items():
            new_value = getattr(learner, key)
            value.append(new_value)

    def add_k_metrics(self, learner: Learner):
        for key, value in self.metrics_k.items():
            logging.debug('Metrics_k for k:{}'.format(str(key)))
            src_value_dict = getattr(learner, key + 's')
            logging.debug('Dict: {}'.format(src_value_dict))
            for k2, val2 in src_value_dict.items():
                logging.debug('Appending to k2:{} value:{}'.format(k2, val2))
                value[k2].append(val2)

    # def get_features(self, learner:Learner, classifier_name):
    #     if classifier_name == 'XGB':
    #         return learner.coef_xgb
    #     elif classifier_name == 'SVM-L-W':
    #         return learner.coef_svc
    #     elif classifier_name == 'LR-W':
    #         return learner.coef_lr
    #     elif classifier_name == 'GBC':
    #         return learner.coef_gbc
    #     elif classifier_name == 'RF':
    #         return learner.coef_rf
    #     elif classifier_name == 'ExtraTrees':
    #         return learner.coef_et
    #     else:
    #         return None

    def get_classifier_results(self, learner: Learner):
        classifier_names = learner.names
        classifier_results = {}
        for name in classifier_names:
            classifier_results[name] = ClassifierResult()
            classifier_results[name].features = learner.features[name]

        for metric in self.metrics:
            classifier_metrics = getattr(learner, metric)
            for index, name in enumerate(classifier_names):
                metric_value = classifier_metrics[index]
                classifier_results[name].metrics[metric] = metric_value

        for metric_k in self.metrics_k:
            src_value_dict = getattr(learner, metric_k)
            for k, metrics in src_value_dict.items():
                for index, name in enumerate(classifier_names):
                    try:
                        metric_value = metrics[index]
                    except IndexError:
                        metric_value = np.nan
                        logging.error("Metric {} not available for k={}".format(name, str(k)))
                    classifier_results[name].metrics_k[metric_k][k] = metric_value

        return classifier_results

    def _get_class_counts(self, data):
        class_counts = {'0': len(np.where(data[self.label_name] < 1)[0]),
                        '1': len(np.where(data[self.label_name] > 0)[0])}
        return class_counts

    def get_class_counts_df(self, relative_counts=True):
        class_list = []
        for m in self.experiment_results:
            list_item = {'day': m.problem_definition.days_to_cutoff,
                         'days_to_predict': m.problem_definition.days_to_predict,
                         'days_for_label_window': m.problem_definition.days_for_label_window,
                         'code_module': m.problem_definition.module,
                         'code_presentation': m.problem_definition.presentation}
            for k, v in m.class_numbers_train.items():
                list_item['train_' + map_k_toclassname(k)] = v
            for k, v in m.class_numbers_test.items():
                list_item['test_' + map_k_toclassname(k)] = v
            class_list.append(list_item)
        df_series = pd.DataFrame(class_list)

        if relative_counts is True:
            df_series.loc[:, 'test_NS_ratio'] = df_series['test_NS'] / (df_series['test_NS'] + df_series['test_S'])
            df_series.loc[:, 'test_S_ratio'] = df_series['test_S'] / (df_series['test_NS'] + df_series['test_S'])
            df_series.loc[:, 'train_NS_ratio'] = df_series['train_NS'] / (df_series['train_NS'] + df_series['train_S'])
            df_series.loc[:, 'train_S_ratio'] = df_series['train_S'] / (df_series['train_NS'] + df_series['train_S'])
            df_series = df_series[
                ['code_module', 'code_presentation', 'day', 'days_to_predict', 'days_for_label_window', 'train_NS_ratio', 'test_NS_ratio',
                 'train_S_ratio', 'test_S_ratio']]

        return df_series

    def get_metrics_df(self):
        metric_list = []

        for r in self.experiment_results:
            for key, cr in r.classifier_results.items():
                list_item = {'day': r.problem_definition.days_to_cutoff,
                             'days_to_predict': r.problem_definition.days_to_predict,
                             'days_for_label_window': r.problem_definition.days_for_label_window,
                             'code_module': r.problem_definition.module,
                             'code_presentation': r.problem_definition.presentation,
                             'classifier': key}
                for key2, metric in cr.metrics.items():
                    list_item[key2] = metric
                metric_list.append(list_item)

        return pd.DataFrame(metric_list)

    def get_metric_daily_df(self, metric_name, k=None):
        if k is None:
            df = self.get_metrics_df()
        else:
            df = self.get_metrics_k_df()
            df = df.loc[df['k'] == k]

        df_metric = df[['day', 'days_to_predict', 'days_for_label_window', 'code_module', 'classifier', metric_name]]
        df_metric = df_metric.set_index(['day', 'days_to_predict', 'days_for_label_window', 'code_module', 'classifier'])
        df_metric = df_metric.unstack().reset_index().groupby(['day']).mean()
        df_metric.columns = df_metric.columns.droplevel(0)
        return df_metric

    def get_features_df(self):
        """
        Gets the DataFrame containing all features
        """
        df = pd.DataFrame()
        for r in self.experiment_results:
            df_part = pd.DataFrame({'feature_name': r.feature_names,
                                    'code_module': r.problem_definition.module,
                                    'code_presentation': r.problem_definition.presentation,
                                    'day': r.problem_definition.days_to_cutoff,
                                    'days_to_predict': r.problem_definition.days_to_predict,
                                    'days_for_label_window': r.problem_definition.days_for_label_window,
                                    'assessment': r.problem_definition.assessment_name
                                    })
            for clf_name, clf_result in r.classifier_results.items():
                if len(clf_result.features) > 0:
                    df_part[clf_name] = clf_result.features
            df = df.append(df_part)
        return df

    def get_top_daily_features(self, classifier, top=10):
        df = self.get_features_df()
        df = df[['code_module', 'day', 'feature_name', classifier]]
        df_g = df.groupby(['day', 'feature_name'], sort=True).mean()
        g = df_g[classifier].groupby(level=0, group_keys=False)
        return g.apply(lambda x: x.order(ascending=False).head(top))

    def plot_metric(self, metric_name, k=None, metric_friendly_name=None, ymin=0, ymax=1, label_postfix='', width=14,
                    height=None):
        df = self.get_metric_daily_df(metric_name, k)
        min_index = df.index.min()
        max_index = df.index.max()
        index_name = df.index.name

        if metric_friendly_name is None:
            metric_friendly_name = metric_name

        plot_df(df, title=metric_friendly_name + ' for ' + index_name + ' ' + str(min_index) + ' to ' + str(
            max_index) + ' ' + label_postfix, width=width, height=height, xlabel=index_name,
                ylabel=metric_friendly_name,
                ymin=ymin)

    def get_metrics_k_df(self):
        df_metrics_k_all = pd.DataFrame()

        for r in self.experiment_results:
            for key, cr in r.classifier_results.items():
                df_metrics_k = pd.DataFrame()

                for metric_name, metric_values in cr.metrics_k.items():
                    metric_k_list = []
                    for k, value in metric_values.items():
                        list_item = {'day': r.problem_definition.days_to_cutoff,
                                     'days_to_predict': r.problem_definition.days_to_predict,
                                     'days_for_label_window': r.problem_definition.days_for_label_window,
                                     'code_module': r.problem_definition.module,
                                     'code_presentation': r.problem_definition.presentation,
                                     'classifier': key,
                                     'k': k,
                                     metric_name: value}
                        metric_k_list.append(list_item)
                    df_metric_k = pd.DataFrame(metric_k_list)
                    try:
                        df_metrics_k = df_metrics_k.merge(df_metric_k,
                                                          on=['code_module', 'code_presentation', 'day',
                                                              'days_to_predict', 'days_for_label_window', 'k',
                                                              'classifier'])
                    except KeyError:
                        df_metrics_k = df_metric_k
                df_metrics_k_all = df_metrics_k_all.append(df_metrics_k)
        return df_metrics_k_all

    def get_metric_k_daily_df(self, metric_name, k):
        df = self.get_metrics_k_df()
        df = df.loc[df['k'] == k, ['day', 'days_to_predict', 'days_for_label_window', 'code_module', 'classifier', 'k', metric_name]]
        # df_metric = df[['day', 'code_module', 'classifier','k', metric_name]]

    def plot_class_counts(self, relative_counts=True, palette="deep"):
        """
        Creates the line plot for the class counts
        :param relative_counts: Specifies whether relative counts should be used.
        :param palette: Palette used for the graph
        """
        df_series = self.get_class_counts_df(relative_counts=relative_counts)
        df_series = pd.melt(df_series, id_vars=['code_module', 'day'],
                            value_vars=['train_NS_ratio', 'test_NS_ratio'], var_name='type', value_name='NS_ratio')
        day_max = df_series['day'].max()
        sns.set_palette(palette)
        ax = sns.plt.axes()
        sns.tsplot(time='day', unit='code_module', condition='type', value='NS_ratio', data=df_series)
        ax.set_title('Ratio of NotSubmit class in training and testing data')
        sns.plt.ylim(0, )
        sns.plt.xlim(0, )
        sns.plt.xticks(np.arange(0, day_max + 1))
        sns.plt.yticks(np.arange(0.0, 1.01, 0.1))
        # plt.savefig("label_ratios.png", bbox_inches='tight', dpi=1000, format='png')
        sns.plt.show()

    def include_modpres_to_df(self, df: pd.DataFrame, problem_definition: ProblemDefinition):
        df['module'] = problem_definition.module
        df['presentation'] = problem_definition.presentation
        return df

    def filter_top_k_columns(self, df: pd.DataFrame, k, column_name):
        return df

    def print_weights(self, feature_col_name='feature_name', top_k_features=10):
        if len(self.feature_extractors) > 0:
            return

        learner_priorities = ['xgb', 'svc', 'lr', 'et', 'gbc']
        day = 1
        df_all_features = pd.DataFrame()
        for learner in self.learners:
            print("DAY: ", day)
            day += 1
            df_new_lr = self.get_coef_df(learner, feature_col_name=feature_col_name)
            for learner_name in learner_priorities:
                learner_coef = learner_name + '_coef'
                if learner_coef in df_new_lr:
                    print(df_new_lr.sort_values(by=[learner_coef], ascending=[True]))
                    df_top = df_new_lr.sort_values(by=learner_coef, ascending=False).head(n=top_k_features)
                    df_top = self.include_modpres_to_df(df_top)
                    df_top['day'] = day
                    print(df_top)
                    df_all_features = pd.concat([df_all_features, df_top])
                    print('all features')
                    print(df_all_features)
                break
        return df_all_features

    def create_problem_definitions(self, module_presentations, assessment_name, min_days, max_days, label_name,
                                   grouping_column,
                                   id_column, training_type, max_days_to_predict, count_all_days_to_predict=False,
                                   days_for_label_window=None, count_all_days_for_label_window=False,
                                   filter_only_registered=True):
        """
        Creates list of problem definitions that are used for experiments. These can be trained separately in parallel.
        :param min_days:
        :param count_all_days_to_predict:
        :param max_days_to_predict:
        :param max_days_to_predict:
        :param module_presentations:
        :param assessment_name:
        :param max_days:
        :param label_name:
        :param grouping_column:
        :param id_column:
        :param training_type:
        :return:
        """
        problem_definitions = []
        days = np.arange(min_days, max_days + 1)
        for (module, presentation_test, presentation_train) in module_presentations:
            for day in days:
                min_days_to_predict = min(max_days_to_predict, day)
                if count_all_days_to_predict:
                    max_days_to_predict_local = np.arange(min_days_to_predict + 1)
                else:
                    max_days_to_predict_local = [min_days_to_predict]

                for days_to_predict in max_days_to_predict_local:
                    if days_for_label_window is None:
                        days_for_label_window = days_to_predict

                    if count_all_days_for_label_window:
                        max_days_for_label = np.arange(days_for_label_window)
                    else:
                        max_days_for_label = [days_for_label_window]

                    for days_for_label_window_local in max_days_for_label:
                        problem_definition = ProblemDefinition(module, presentation_test, assessment_name,
                                                               days_to_cutoff=day,
                                                               days_to_predict=days_to_predict,
                                                               days_for_label_window=days_for_label_window_local,
                                                               filter_only_registered=filter_only_registered,
                                                               y_column=label_name, grouping_column=grouping_column,
                                                               id_column=id_column, presentation_train=presentation_train,
                                                               training_type=training_type)
                        problem_definitions.append(problem_definition)
        return problem_definitions
