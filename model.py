import pandas as pd
import pandas_profiling
import numpy as np
import datetime
import matplotlib.pyplot as plt
from itertools import permutations
import tqdm
import random
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

random.seed(0)

def train_model(df, start_date, save_model = True):

    names = df.columns.values
    X = df[names[:-1]]
    y = pd.DataFrame({'y': df[names[-1]]})

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=1234)

    pd.DataFrame.to_csv(X_test, 'X_test.csv', index=False)

    X_train['date'] = X_train['date'].apply(
        lambda x: (start_date - datetime.datetime.fromisoformat(x)).total_seconds() / 3600.)
    X_test['date'] = X_test['date'].apply(
        lambda x: (start_date - datetime.datetime.fromisoformat(x)).total_seconds() / 3600.)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.6, random_state=1234)
    print(X_train.shape, X_validation.shape, X_test.shape)
    print(y_train.shape, y_validation.shape, y_test.shape)

    best_model = CatBoostRegressor(
        logging_level='Silent',
        custom_metric='MAPE',
        # bootstrap_type='Bernoulli',
        # subsample=1,
        # bagging_temperature=0.00000001,
        #  sampling_frequency='PerTree',
        depth=12,
        # random_strength=1,
        border_count=251,
        # feature_border_type='Uniform',
        # thread_count=3,
        iterations=500,
        # l2_leaf_reg = 100,
        learning_rate=1,
        # save_snapshot=True,
        # snapshot_file='snapshot_best.bkp',
        random_seed=63,
        od_type='Iter',
        od_wait=20,
        loss_function='MAE',
        use_best_model=True
        # leaf_estimation_method='Gradient'
        # grow_policy='Depthwise',
        # max_leaves=160
        # min_data_in_leaf=30
        # score_function='NewtonCosine'
        # boost_from_average=False
        # boosting_type='Ordered'
    )
    # RMSE, MAE, MAPE
    best_model.fit(X_train, y_train,
                   eval_set=(X_validation, y_validation),
                   plot=True)

    print('Resulting tree count:', best_model.tree_count_)

    y_pred = best_model.predict(X_test)
    y_test_arr = y_test.values
    y_test_arr = y_test_arr.reshape(y_pred.shape)

    sorted_pred_diff = np.sort(np.abs(y_test_arr - y_pred) / y_test_arr)

    print('MAPE median =', sorted_pred_diff[y_pred.size // 2])
    print('MAPE mean =', (np.abs(y_test_arr - y_pred) / y_test_arr).mean())

    if save_model:
        best_model.save_model('best_model')

    return (np.abs(y_test_arr - y_pred) / y_test_arr).mean()

if __name__ == '__main__':
    df = pd.read_csv('ds1.csv', header=0)
    start_date = datetime.datetime.fromisoformat('2022-01-01 08:59:59.999')
    train_model(df, start_date)
    # print('prediction')
    # best_model = CatBoostRegressor()
    # best_model.load_model('best_model')
    # df = pd.read_csv('X_test.csv', header=0)
    # # print(df.columns.values)
    # start_date = datetime.datetime.fromisoformat('2021-01-01 08:59:59.999')
    # df['date'] = df['date'].apply(lambda x: (start_date - datetime.datetime.fromisoformat(x)).total_seconds() / 3600.)
    #
    # prediction = best_model.predict(df)
    # print(prediction)