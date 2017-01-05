from pandas import DataFrame


def print_df(df: DataFrame, formats=None):
    """
    Prints the dataframe on the output for the specified list of formats
    :param df:
    :param formats: one of ['tab','latex']
    """
    if formats is None:
        formats = ['tab']
    if 'tab' in formats:
        print(df)
    if 'latex' in formats:
        print(df.to_latex(float_format=lambda x: '%.2f' % x))


def get_prev_pres_same(pres):
    if pres == '2014J':
        return '2013J'
    elif pres == '2014B':
        return '2013B'
    elif pres == '2013J':
        return '2012J'
    elif pres == '2013B':
        return '2012B'
    else:
        return None

def get_prev_pres_closest(pres):
    if pres == '2014J':
        return '2014B'
    elif pres == '2014B':
        return '2013J'
    elif pres == '2013J':
        return '2013B'
    elif pres == '2013B':
        return '2012J'
    else:
        return None