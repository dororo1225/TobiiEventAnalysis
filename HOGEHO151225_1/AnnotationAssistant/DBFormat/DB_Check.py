# -*- coding: utf-8 -*-
__author__ = 'yamamoto'

import sqlite3
import datetime
import sys
import os

# USBタグ(FW1, FW2 etc.)
obs_id = os.getcwd().split('\\')[-3]

db_name = obs_id  + '.db'
data_path = os.getenv("HOMEDRIVE") + \
                    os.getenv("HOMEPATH") +  \
                    "\\Dropbox\\AnnotationAssistant\\" + db_name

# db_nameの中身を取得
print "-------------------------------------------------------"
print "Check content of " + db_name
print data_path
print "-------------------------------------------------------"
connector = sqlite3.connect(data_path)
cursor = connector.cursor()

# samplesテーブルのレコードを取得
cursor.execute("select * from samples")
for var in range(0, 30):  # 先頭5行のレコードを表示
    row = cursor.fetchone()
    print row

#for row in cursor:  # すべてのレコードを表示する.
#    print row

# progressテーブルのレコードを取得
cursor.execute("select * from progress")
for row in cursor:
    print row

# sample.dbを閉じる
connector.close()