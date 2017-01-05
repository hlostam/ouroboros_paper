from pandas import DataFrame
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def precision_at_k(y_true, y_score, k):
    df_sorted = DataFrame({'score': y_score.tolist(), 'real_value': y_true.tolist()}).sort_values(by='score')
    len_abs = k * len(df_sorted)
    len_abs_int = int(len_abs)
    threshold = df_sorted.iloc[len_abs_int, 1]
    y_pred = [1 if i >= threshold else 0 for i in df_sorted['score']]
    return precision_score(df_sorted['real_value'], y_pred)


def recall_at_k(y_true, y_score, k):
    df_sorted = DataFrame({'score': y_score.tolist(), 'real_value': y_true.tolist()}).sort_values(by='score')
    len_abs = k * len(df_sorted)
    len_abs_int = int(len_abs)
    threshold = df_sorted.iloc[len_abs_int, 1]
    y_pred = [1 if i >= threshold else 0 for i in df_sorted['score']]
    return recall_score(df_sorted['real_value'], y_pred)

def top_k_precision(y_true, y_score, k):
    """
    Counts the precision for given y_score and real values in the top-k instances ordered by the probability of being
    of the target class.
    :param y_score:
    :param y_true:
    :param k:
    :return:
    """
    k_pct = (100 - k) / 100.
    return precision_at_k(y_true, y_score, k_pct)


def top_k_recall(y_true, y_score, k):
    """
    Counts the precision for given y_score and real values in the top-k instances ordered by the probability of being
    of the target class.
    :param y_score:
    :param y_true:
    :param k:
    :return:
    """
    k_pct = (100 - k) / 100.
    return recall_at_k(y_true, y_score, k_pct)


def main():
    y_score = np.array([0.4, 0.8, 0.1, 0.2, 0.5])
    y_true = np.array([1., 1., 0., 0., 1.])
    for k in range(5, 101, 5):
        print(k, top_k_precision(y_true, y_score, k), top_k_recall(y_true, y_score, k))
        
if __name__ == "__main__":
    main()
