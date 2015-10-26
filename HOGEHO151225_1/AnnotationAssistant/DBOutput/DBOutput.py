# -*- coding: utf-8 -*-
__author__ = 'yamamoto'

import os
import sqlite3
import pandas as pd

################################################
#  処理関数
################################################
def db_output(file_name):
    obs_id = file_name.split('.db')[0]

    positive = open('info.txt', 'a')
    negative = open('bg.txt', 'a')

    Dpos = []
    # sample.dbの中身を取得
    connector = sqlite3.connect(file_name)
    cursor = connector.cursor()

    # samplesテーブルのレコードを取得
    cursor.execute("select * from samples")
    for row in cursor:
        print row
        if row[6] == 200:  # positive data
            coord = ' '.join([str(int(e)) for e in row[2:6]])
            s1 = '%s %s %s\n' % (row[1], str(int(1)), coord)
            positive.write(s1)
            Dpos.append(row)
        else:  # negative data or not coded
            s = '%s\n' % row[1]
            negative.write(s)

    # positiveデータ・ネガティブデータ用のファイルを閉じる
    negative.close()
    positive.close()

    if not len(Dpos) == 0:
        df = pd.DataFrame(Dpos)
        df.columns = ["id", "imgpath", "x", "y", "width", "height", "status", "name", "datetime"]
        print df
        if not os.path.isfile('AnnotationData.csv'):
            df.to_csv('AnnotationData.csv', mode='a', index=False)
        else:
            df.to_csv('AnnotationData.csv', mode='a', index=False,  header=False)

    # progressテーブルのレコードを取得
    cursor.execute("select * from progress")
    for row in cursor:
        print row
        progress = row[0]
        img_total = row[1]
    connector.close()  # dbを閉じる

    progress = [(img_total, progress, 100 * progress / img_total, len(df), progress - len(df))]
    df_progress = pd.DataFrame(progress)
    df_progress.columns = ["total", "progress", "percent", "positive", "negative"]
    df_progress["obs_id"] = obs_id
    print df_progress
    if not os.path.isfile("progress.csv"):
        df_progress.to_csv('progress.csv', mode='a', index=False)
    else:
        df_progress.to_csv('progress.csv', mode='a', index=False,  header=False)


################################################
#  メイン処理
################################################

obs_id = os.getcwd().split('\\')[-3]

db_name = obs_id  + '.db'
data_path = os.getenv("HOMEDRIVE") + \
                    os.getenv("HOMEPATH") +  \
                    "\\Dropbox\\AnnotationAssistant\\" + db_name

db_output(data_path)