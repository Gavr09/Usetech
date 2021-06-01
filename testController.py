from flask import Flask, request
import pandas as pd
from model import train_model
import datetime
from catboost import CatBoostRegressor
app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def test():
    f = request.files['the_file']

    file_contents = f.stream.read().decode("utf-8")
    content_list = file_contents.split('\n')
    col_names = content_list[0].split(',')
    col_num = len(col_names)
    my_dict = {}
    for key in col_names:
        my_dict[key] = []

    for i in range(len(content_list)-2):
        line = content_list[i+1].split(',')
        for j in range(col_num):
            if j == 0:
                my_dict[col_names[j]].append(line[j])
            else:
                my_dict[col_names[j]].append(float(line[j]))

    df = pd.DataFrame(my_dict)

    start_date = datetime.datetime.fromisoformat('2022-01-01 08:59:59.999')

    if 'y' in col_names:
        print('train')
        res = train_model(df, start_date, save_model = False)
        print(res)
        return str(res)
    else:
        print('prediction')
        best_model = CatBoostRegressor()
        best_model.load_model('best_model')
        df['date'] = df['date'].apply(
            lambda x: (start_date - datetime.datetime.fromisoformat(x)).total_seconds() / 3600.)
        prediction = best_model.predict(df)
        return str(prediction)


if __name__ == '__main__':
    app.run()