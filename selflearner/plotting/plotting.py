import os

import itertools

__author__ = 'author'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cmx
import matplotlib.colors as colors

plotting_path_base = "plots"
# class Plotting:
#     self.plotting_directory = os.path.join(os.sep, os.path.dirname(__file__), self.data_directory,
#                                               problem_definition.string_representation)

def get_plotting_dir_checked():
    plotting_dir = os.path.join(os.sep, os.path.dirname(__file__), plotting_path_base)
    if not os.path.exists(plotting_dir):
        os.makedirs(plotting_dir)
    return plotting_dir


def get_plotting_path(file_name):
    return os.path.join(get_plotting_dir_checked(), file_name)


def plot_confusion_matrix(cm, target_names, file_prefix, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig( get_plotting_path('_'.join([file_prefix, 'confusion_matrix.png']),
                 format='png', dpi=1000) )


def plot_metric_lines(values, clf_names, metric_name, file_prefix):
    plt.figure()
    data = np.array(values)
    days = np.arange(data.shape[0])

    for clf_id in range(len(clf_names)):
        plt.plot(days, data[:, clf_id], label=clf_names[clf_id], )

    ax = plt.gca()
    ax.xaxis.grid(True)
    plt.xticks(days)
    plt.xlabel('Days to cutoff')
    plt.ylabel(metric_name)
    plt.ylim([0.1, 1.0])
    plt.title(metric_name + ' for classifiers for various days to cutoff')

    plt.legend(loc="lower left")
    plt.savefig(get_plotting_path('_'.join([file_prefix, metric_name + '.png'])), format='png', dpi=1000)


def plot_auc_heatmap(aucs, clf_names, file_prefix):
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    data = np.array(aucs)
    rows = np.arange(data.shape[0])

    ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

    plt.pcolor(data, cmap=plt.cm.Reds, edgecolors='k')

    ax.set_xticklabels(clf_names, minor=False)
    ax.set_yticklabels(rows, minor=False)

    plt.xticks(rotation=90)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.savefig(get_plotting_path('_'.join([file_prefix, 'auc_heatmap.png'])), format='png', dpi=1000)


def plot_df_heatmap(df_pivoted, measure_name, file_prefix):
    plt.figure()
    sns.set()
    sns_plot = sns.heatmap(df_pivoted, annot=True)
    plt.savefig(get_plotting_path('_'.join([file_prefix, measure_name, 'heatmap.png'])), format='png', dpi=1000)


def plot_pr_curve(pr_list, file_prefix, days_to_cutoff):
    plt.figure()
    for name, list in pr_list.items():
        for pair in list:
            plt.plot(pair[0], pair[1], label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Day: %s Precision-Recall' % str(file_prefix))
    plt.legend(loc="lower left")
    plt.savefig(get_plotting_path('_'.join([file_prefix, 'pr_curve.png'])), format='png', dpi=1000)


def plot_summary(df_orig, metric_name, file_prefix):
    plt.figure()

    df_orig = df_orig.reset_index()
    if 'BagTree' in df_orig.columns:
        df_orig = df_orig.drop(['BaggTree'], axis=1)
    if 'ExtraTrees' in df_orig.columns:
        df_orig = df_orig.drop(['ExtraTrees'], axis=1)
    df = df_orig.iloc[:, 1:]
    print('---')
    print(len(df.columns))
    print(df)

    plt.style.use('ggplot')
    fontP = FontProperties()
    fontP.set_size('small')

    values = range(len(df.columns))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    jet = plt.get_cmap('jet')
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for col in range(len(df.columns)):
        colorVal = scalarMap.to_rgba(values[col])
        plt.plot(df_orig.iloc[:,0], df.iloc[:, col], linewidth=1, label=df.columns[col], color=colorVal)
    ax = plt.gca()
    ax.xaxis.grid(True)
    plt.xticks(df_orig.iloc[:,0])
    plt.xlabel('Days to cutoff')
    plt.ylabel(metric_name)
    plt.ylim([0.1, 1.0])
    plt.title(metric_name + ' for classifiers for various days to cutoff')
    plt.legend(loc="best", prop=fontP)
    plt.savefig(get_plotting_path('_'.join([file_prefix, metric_name, '.png'])), format='png', dpi=1000)


def plot_df(df, title='', ymin=0, width=14, height=None, ylabel='', xlabel=''):
    #     plt.clf()
    sns.set_context("paper")
    sns.set_palette("deep")
    if height is None:
        height = width / 2
    font_size = height * 2.1
    markers = itertools.cycle(('+', '.', 'o', '*', 'v', 's', 'D', 'h', '^', 'p'))

    plt.style.use('seaborn-whitegrid')
    params = {
        'axes.labelsize': font_size,
        'font.size': font_size,
        'text.fontsize': font_size,
        'legend.fontsize': font_size * 1.1,
        'xtick.labelsize': font_size * 1.1,
        'ytick.labelsize': font_size * 1.1,
        'figure.figsize': (width, height),
        'lines.marker': None,
        'axes.linewidth': 1.0,
        'lines.linewidth': width / 7,
        'legend.frameon': True,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
    }
    plt.rcParams.update(params)

    plt.xticks(df.index)
    plt.ylim(ymin, 1.0)
    for i, c in enumerate(df.columns):
        plt.plot(df[c], label=c, marker=next(markers), markersize=width * 0.7)
    plt.title(title, fontsize=font_size * 1.4)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc=1, ncol=4, fancybox=True)
    plt.show()