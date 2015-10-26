# -*- coding: utf-8 -*-
__author__ = 'yamamoto'

import sqlite3
import sys
import os

# DBのフォーマット関数
def format_db(dpath, fpath):
    conn = sqlite3.connect(dpath)
    # samplesテーブル
    sql = u"""
    create table samples (
    id integer primary key autoincrement,
    filepath text unique not null,
    x integer not null default 0,
    y integer not null default 0,
    width integer not null default 0,
    height integer not null default 0,
    status integer not null default 0,
    teacher text default 'no name',
    updated_date timestamp default (datetime('now', 'localtime'))
    );
    """
    conn.execute(sql)

    sql = u"""
    create index id_idx on samples(id);
    """
    conn.execute(sql)

    sql = u"""
    create table progress (
    pos integer primary key,
    total integer not null
    );
    """
    conn.execute(sql)

    sql = u"""
    create index pos_idx on progress(pos);
    """
    conn.execute(sql)

    for item in fpath:
        sql = u"""insert into samples(filepath, status) values (?, 100); """
        conn.execute(sql, (unicode(item, sys.getfilesystemencoding()),))

    sql = u"""insert into progress values (?,?);"""
    fnum = (0, len(fpath))
    conn.execute(sql, fnum)
    conn.commit()
    conn.close()


obs_id = os.getcwd().split('\\')[-3]

# DBに登録する画像データ一覧を取得
# DBF_dir = os.getcwd()  # 現在の作業ディレクトリを取得
# os.chdir('..')  # 1つ上のディレクトリを移動
os.chdir('../FaceClipper/static/images')
files = os.listdir(os.getcwd())  # ディレクトリ内のファイル名を取得

# 画像データのpathのリストを作成
file_path = []
for var in files:
    file_path.append('static/images/' + var)
# file_num = (0, len(file_path))

os.chdir('../../../DBFormat')  # デバッグ初期の作業ディレクトリに戻す
print os.getcwd()

# DBファイルの場所としてDropboxのpathを取得
db_name = obs_id + '.db'
data_path = os.getenv("HOMEDRIVE") + \
                    os.getenv("HOMEPATH") +  \
                    "\\Dropbox\\AnnotationAssistant\\" + db_name
print data_path


if os.path.isfile(data_path):
    print "---------------------------------------------"
    print db_name + " has already existed in Dropbox."
    print "Check contents of 'samples.db'"
    print "---------------------------------------------"

    data_path = os.getcwd() + "\\samples.db"

    if os.path.isfile(data_path):
        print "---------------------------------------------"
        print db_name + " has already existed in DBFormat."
        print "Check contents of 'samples.db'"
        print "---------------------------------------------"
    else:
        format_db(data_path, file_path)
else:
    format_db(data_path, file_path)


# sample.dbの中身を取得
print "-------------------------------------------------------"
print "Check content of " + db_name
print data_path
print "-------------------------------------------------------"
connector = sqlite3.connect(data_path)
cursor = connector.cursor()

# samplesテーブルのレコードを取得
cursor.execute("select * from samples")
for var in range(0, 5):  # 先頭5行のレコードを表示
    row = cursor.fetchone()
    print row
# for row in cursor:  # すべてのレコードを表示する.
# print row

# progressテーブルのレコードを取得
cursor.execute("select * from progress")
for row in cursor:
    print row

# sample.dbを閉じる
connector.close()