
def get_expected_submission(module, presentation):
    dic =   {('AAA', '2013J'): 83,
             ('AAA', '2014J'): 84,
             ('BBB', '2013B'): 472,
             ('BBB', '2013J'): 377,
             ('BBB', '2014B'): 266,
             ('BBB', '2014J'): 369,
             ('CCC', '2014B'): 910,
             ('CCC', '2014J'): 1114,
             ('DDD', '2013B'): 327,
             ('DDD', '2013J'): 380,
             ('DDD', '2014B'): 273,
             ('DDD', '2014J'): 345,
             ('EEE', '2013J'): 216,
             ('EEE', '2014B'): 146,
             ('EEE', '2014J'): 283,
             ('FFF', '2013B'): 298,
             ('FFF', '2013J'): 333,
             ('FFF', '2014B'): 261,
             ('FFF', '2014J'): 516,
             ('GGG', '2013J'): 219,
             ('GGG', '2014B'): 205,
             ('GGG', '2014J'): 212}

    return int(dic[module, presentation])
