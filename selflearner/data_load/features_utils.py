import pandas as pd


class FeaturesMapping:
    df_imd_mapping = pd.DataFrame({
        'imd_value': [.05,.15,.25,.35,.45,.55,.65,.75,.85,.95, None],
    },
        index=['0-10%', '10-20', '20-30%', '30-40%', '40-50%', '50-60%','60-70%','70-80%','80-90%','90-100%', None]
    )

    def map_imd_band(self, df_mapped):
        """
        Maps the data frame with IMD band strings to their mid-range values, i.e. 0-10 is mapped as 0.05 etc.
        :param df_mapped: DataFrame
        :return:
        """
        return pd.merge(self.df_imd_mapping, df_mapped, left_index=True, right_on='imd_band')

    @staticmethod		
    def consecut(x, start=1):
        """
        Counts the number of consecutive numbers in the list starting with the given start index, the default value = 1
        :param x:
        :param start:
        :return:
        """
        to_find = start
        for item in x:
            if item == to_find:
                to_find = to_find + 1
            else:
                break
        return to_find - 1

    @staticmethod
    def result_to_pass_fail(row):
        if row['final_result'] == 'Distinction':
            return 'Pass'
        if row['final_result'] == 'Pass':
            return 'Pass'
        return 'Fail'
