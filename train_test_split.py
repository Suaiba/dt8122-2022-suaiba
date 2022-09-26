import pandas as pd
import numpy as np
def get_train_test(csv_file):
    df = pd.read_csv(csv_file)
    x=df['feature_1']
    y=df['feature_2']
    N = len(df)
    x_train=[]
    y_train=[]
    x_test = []
    y_test = []
    indices = np.random.RandomState(seed=10).permutation(np.arange(0, N))
    indices_train = indices[0:int(N * 0.8)]
    indices_test = indices[int(N * 0.8):]
    for i in indices_train:
        x_train.append(x[i])
        y_train.append(y[i])
    for j in indices_test:
        x_test.append(x[i])
        y_test.append(y[i])

    df2 = pd.DataFrame(list(zip(x_train, y_train)), columns=['feature_1', 'feature_2'])
    df3 = pd.DataFrame(list(zip(x_test, y_test)), columns=['feature_1', 'feature_2'])

    return df2,df3