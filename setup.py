target_col = ['exited']
independent_cols = ['lastmonth_activity',
                    'lastyear_activity', 'number_of_employees']


def split_features_target(dataframe):
    x = dataframe[independent_cols].values.reshape(-1,len(independent_cols))
    y = dataframe[target_col].values.reshape(-1,len(target_col))

    return x, y