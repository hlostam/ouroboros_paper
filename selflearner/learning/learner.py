from selflearner.problem_definition import ProblemDefinition

__author__ = 'author@gmail.com'

import logging

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, average_precision_score, \
    recall_score, precision_score
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import linear_model as LR
from sklearn import preprocessing

from .evaluation_util import top_k_precision
from .evaluation_util import top_k_recall
from selflearner.plotting import plotting


class Learner:
    def __init__(self, train_data, test_data, label, problem_definition: ProblemDefinition, sampler=None,
                 classifiers=None, feature_extractors=None, topk_prec_list=None, optimise_threshold=True,
                 plot_pr_curve=False):
        if topk_prec_list is None:
            topk_prec_list = [5, 10, 20, 50, 75]
        if feature_extractors is None:
            feature_extractors = []
        self.auc = []
        self.fscore = []
        self.prec = []
        # self.top_k_prec = []
        self.recall = []
        self.pr_auc = []
        # self.coef_lr = None
        # self.coef_svc = None

        self.optimise_threshold = optimise_threshold
        self.plot_pr_curve = plot_pr_curve

        self.problem_definition = problem_definition
        logging.info("Days to cutoff: %s" % problem_definition.days_to_cutoff)
        self.train_data = train_data
        self.test_data = test_data
        self.label = problem_definition.y_column
        self._init_classifiers(classifiers)
        self.sampler = sampler
        self.feature_extractors = feature_extractors
        self.topkpreclist = topk_prec_list
        self.top_k_prec = {}
        self.top_k_rec = {}
        for k in self.topkpreclist:
            self.top_k_prec[k] = []
            self.top_k_rec[k] = []
        self.features = {}
        for clf_name in self.names:
            self.features[clf_name] = []

    def _init_classifiers(self, classifiers):
        if classifiers is None:
            classifiers = [
                # (OneClassSVM(kernel="linear" ), "SVM-One-L"),
                # (OneClassSVM(kernel="rbf", nu=0.1, gamma=0.1), "SVM-One-R"),

                # (KNeighborsClassifier(10), "kNN 7"),
                (SVC(kernel="linear", C=0.025, probability=True), "SVM-L"),
                (SVC(kernel="linear", C=0.025, probability=True, class_weight="balanced"), "SVM-L-W"),
                (SVC(kernel="rbf", gamma=0.1, C=1, probability=True, class_weight="balanced"), "SVM-W-R"),
                # (SVC(gamma=2, C=1, probability=True),"kNN 5")
                # (tree.DecisionTreeClassifier(max_depth=5), "DT"),
                # (RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), "RF"),
                # (AdaBoostClassifier(), "AdaBoost"),
                # (GaussianNB(), "gNB"),
                # (BernoulliNB(alpha=.01), "bNB"),
                # (LDA(), "LDA"),
                # (LR.LogisticRegression(C=1e5, penalty='l1'), "LR-L1"),
                (LR.LogisticRegression(C=1e5, penalty='l2'), "LR"),
                (LR.LogisticRegression(C=1e5, penalty='l1', class_weight="balanced"), "LR-W"),
                # (DummyClassifier(strategy="stratified"), "BaseStrat"),
                # (BaggingClassifier(), "BaggTree"),
                # (RandomForestClassifier(class_weight="balanced"), "RF"),
                (GradientBoostingClassifier(), "GBC"),
                # (ExtraTreesClassifier(), "ExTree"),
                # (ExtraTreesClassifier(class_weight="balanced"), "ExTree-W"),
                (DummyClassifier(strategy="constant", constant=0), "Base[S]")
                # (DummyClassifier(strategy="constant", constant=1), "Base[NS]")
            ]
        self.classifiers = classifiers
        self.names = [name for clf, name in self.classifiers]

    def preprocess(self):
        logging.debug("Pre-processing data...")

        # Same train and test - union of train/test features.
        x_train_cols = self.train_data.columns
        x_test_cols = self.test_data.columns
        x_intersect_cols = [x for x in x_train_cols if x in x_test_cols]
        only_train_cols = [x for x in x_train_cols if x not in x_test_cols]
        only_test_cols = [x for x in x_test_cols if x not in x_train_cols]
        logging.debug("Train columns: %s Test columns: %s", len(self.train_data.columns), len(self.test_data.columns))
        logging.info("Only train: %s", only_train_cols)
        logging.info("Only test: %s", only_test_cols)

        self.test_data = self.test_data[x_intersect_cols]
        self.train_data = self.train_data[x_intersect_cols]
        logging.debug("Train columns: %s Test columns: %s", len(self.train_data.columns), len(self.test_data.columns))

        drop_cols = [self.problem_definition.y_column, self.problem_definition.id_column,
                     self.problem_definition.grouping_column]

        x_num_train = self.train_data.select_dtypes(include=['int64', 'float']).drop(drop_cols, axis=1)
        col_names_num = x_num_train.columns.values
        x_num_train = x_num_train.as_matrix()

        x_num_test = self.test_data.select_dtypes(include=['int64', 'float']).drop(drop_cols, axis=1).as_matrix()
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        x_num_train = imp.fit_transform(x_num_train)
        x_num_test = imp.fit_transform(x_num_test)

        # # scale to <0,1>
        # max_train = np.amax(x_num_train, 0)
        # print(max_train)
        # print("IsNaN train:")
        # print(np.any(np.isnan(x_num_train)))

        min_max_scaler = preprocessing.MinMaxScaler()
        x_num_train = min_max_scaler.fit_transform(x_num_train)
        x_num_test = min_max_scaler.fit_transform(x_num_test)  # scale test by max_train

        cat_train = self.train_data.select_dtypes(include=['object'])
        drop_cols = np.intersect1d(drop_cols, cat_train.columns.values)
        cat_train = cat_train.drop(drop_cols, axis=1)

        cat_test = self.test_data.select_dtypes(include=['object'])
        cat_test = cat_test.drop(drop_cols, axis=1)

        cat_train.fillna('NA', inplace=True)
        cat_test.fillna('NA', inplace=True)

        x_cat_train = cat_train.T.to_dict().values()
        x_cat_test = cat_test.T.to_dict().values()

        # vectoring
        vectorizer = DV(sparse=False)
        vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
        vec_x_cat_test = vectorizer.transform(x_cat_test)
        col_names_cat = np.asarray(vectorizer.get_feature_names())
        self.col_names = np.hstack((col_names_num, col_names_cat))

        self.x_train = np.hstack((x_num_train, vec_x_cat_train))
        self.x_test = np.hstack((x_num_test, vec_x_cat_test))

        # HACK: This should be treated in a better way!!!!
        self.y_train = 1. - self.train_data[self.label]
        self.y_test = 1. - self.test_data[self.label]

        logging.info('Train 0: %s', len(np.where(self.y_train < 1)[0]))
        logging.info('Train 1: %s', len(np.where(self.y_train > 0)[0]))
        logging.info('Test 0: %s', len(np.where(self.y_test < 1)[0]))
        logging.info('Test 1: %s', len(np.where(self.y_test > 0)[0]))

    def handle_sampling(self):
        if (self.sampler is not None):
            print("Sampling:{}".format(type(self.sampler).__name__))
            self.x_train, self.y_train = self.sampler.fit_sample(self.x_train, self.y_train)

    def set_classifiers(self, classifiers):
        """Trains the given classifiers"""
        self.classifiers = classifiers
        self.names = [name for clf, name in self.classifiers]

    def get_metrics_for_thr(self, y_prob, y_actual, thr):
        pred = [1. if v > thr else 0. for v in y_prob]
        f_score = f1_score(y_actual, pred)
        prec = precision_score(y_actual, pred)
        recall = recall_score(y_actual, pred)
        conf_matrix = confusion_matrix(y_actual, pred)
        return f_score, prec, recall, conf_matrix

    def find_best_threshold(self, y_prob, y_actual, thresholds):
        """
        Finds the best threshold according  to maximal F-Measure.

        :param thresholds:
        :param y_prob:
        :param y_actual:
        :return:
        """
        max_fscore = 0
        max_thr = 0
        max_prec = 0
        max_recall = 0
        max_conf_matrix = {}

        for t in thresholds:
            fs, pr, rec, cm = self.get_metrics_for_thr(y_prob, y_actual, t)
            if fs > max_fscore:
                max_fscore, max_thr, max_prec, max_recall = fs, t, pr, rec
                max_conf_matrix = cm

        return max_thr, max_fscore, max_prec, max_recall, max_conf_matrix

    def train_inner(self):
        """
        Trains all the classifiers in self.classifiers.
        :return:
        """
        self.all_p = np.zeros((len(self.y_test), len(self.classifiers)), dtype=np.float64)

        current_index = 0
        pr_auc_list = {}

        # Feature extraction
        self.extractor_models = []
        for feature_extractor in self.feature_extractors:
            self.x_train = feature_extractor.fit_transform(self.x_train)
            self.x_test = feature_extractor.transform(self.x_test)

        # Sampling
        self.handle_sampling()

        for clf, clf_name in self.classifiers:
            logging.debug("Training " + clf_name)
            pr_auc_list[clf_name] = []

            if hasattr(clf, 'never_logged_column_name'):
                try:
                    clf.never_logged_column_index = np.where(self.col_names == clf.never_logged_column_name)[0][0]
                except IndexError:
                    raise AttributeError("VLE statistics are needed if BaseLine[Active] classifier is used")

            if type(clf).__name__ == "OneClassSVM":
                print("OneClassSVM train")
                x_train = self.x_train[self.y_train < 1]
                clf.fit(x_train)
            else:
                try:
                    clf.fit(self.x_train, self.y_train)
                except ValueError:
                    print("Classifier hasn't been trained, only negative class was provided.")
                    clf = DummyClassifier(strategy="constant", constant=1)
                    clf.fit(self.x_train, self.y_train)

            is_p_ok = False
            if hasattr(clf, 'predict_proba'):
                p = clf.predict_proba(self.x_test)
                try:
                    y_prob = p[:, 1]
                    is_p_ok = True
                except IndexError:
                    print("no positive class in predicted data")
                    is_p_ok = False

            if is_p_ok:
                self.all_p[:, current_index] = y_prob
                current_index += 1

                prec, rec, thresh = precision_recall_curve(self.y_test, y_prob)
                pr_auc_list[clf_name].append((rec, prec))

                # reset for the Base classifiers / don't allow spurious results
                pred = clf.predict(self.x_test)
                max_fscore = f1_score(self.y_test, pred)
                max_prec = precision_score(self.y_test, pred)
                max_recall = recall_score(self.y_test, pred)
                # max_conf_matrix = confusion_matrix(self.y_test, pred)

                # find max threshold if not BaseLine model
                y_proba_for_opt = clf.predict_proba(self.x_train)
                y_proba_for_opt = y_proba_for_opt[:, 1]
                y_actual_for_opt = self.y_train
                if self.optimise_threshold is True:
                    if not clf_name.startswith('Base['):
                        max_thr, max_fscore, max_prec, max_recall, max_conf_matrix = self.find_best_threshold(
                            y_proba_for_opt,
                            y_actual_for_opt,
                            thresh)
                        max_fscore, max_prec, max_recall, max_conf_matrix = self.get_metrics_for_thr(y_prob,
                                                                                                     self.y_test,
                                                                                                     max_thr)
                        logging.debug("Max Thr: {} F1:{}".format(max_thr, max_fscore))

                top_k = 5
                for k in self.topkpreclist:
                    logging.debug('Appending: {}'.format(str(k)))
                    top_k_prec = top_k_precision(self.y_test, y_prob, k)
                    self.top_k_prec[k].append(top_k_prec)
                    top_k_rec = top_k_recall(self.y_test, y_prob, k)
                    self.top_k_rec[k].append(top_k_rec)
                # self.top_k_prec.append(self.top_k_precs[top_k])

                self.fscore.append(max_fscore)
                self.prec.append(max_prec)
                self.recall.append(max_recall)
                pr_auc = average_precision_score(self.y_test, y_prob)

                self.pr_auc.append(pr_auc)

                # cm = confusion_matrix(self.y_test, predictions)
                # print(cm)

                fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_prob)
                logging.debug("AUC-ROC:" + str(metrics.auc(fpr, tpr)))
                logging.debug("AUC-PR:" + str(pr_auc))
                logging.debug("Precision:" + str(max_prec))
                logging.debug("Precision-TOP5:" + str(self.top_k_prec[top_k]))
                logging.debug("Recall-TOP5:" + str(self.top_k_rec[top_k]))
                logging.debug("Recall:" + str(max_recall))
                # logging.debug("Conf. matrix:\n" + str(max_conf_matrix))
                self.auc.append(metrics.auc(fpr, tpr))
            else:
                predictions = clf.predict(self.x_test)
                if type(clf).__name__ == "OneClassSVM":
                    predictions = [1 if x < 0 else 0 for x in predictions]

                f_score = f1_score(self.y_test, predictions)
                logging.info("Original Fscore: {}".format(f_score))
                self.fscore.append(f_score)
                self.prec.append(precision_score(self.y_test, predictions))
                self.recall.append(recall_score(self.y_test, predictions))
                self.pr_auc.append(0)
                self.auc.append(0)
            if hasattr(clf, 'best_params_'):
                print('Best params on train data')
                print(clf.best_params_)
                print("Grid scores on development set:")
                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))

            self.handle_model_features(clf, clf_name)
            # file_prefix =  '_'.join([self.problem_definition.module, self.problem_definition.presentation, self.problem_definition.days_to_cutoff])

        file_prefix = "_".join([self.problem_definition.module, self.problem_definition.presentation,
                                self.problem_definition.assessment_name, str(self.problem_definition.days_to_cutoff)
                                ])
        if len(self.classifiers) > 0 and self.plot_pr_curve == True:
            plotting.plot_pr_curve(pr_auc_list, file_prefix, self.problem_definition.days_to_cutoff)

    def handle_model_features(self, clf, name):
        try:
            if hasattr(clf, 'coef_'):
                self.features[name] = clf.coef_.flatten()
            elif hasattr(clf, 'feature_importances_'):
                self.features[name] = clf.feature_importances_
        except ValueError:
            pass

    def train_all(self):
        """
        Trains all the default classifiers and draws the Precision-Recall plot for them.
        :return:
        """
        self.train_inner()

    def run(self):
        """
        Runs the preprocessing and training of all the classifiers.
        :return:
        """
        self.preprocess()
        self.train_all()
