# -*- coding: utf-8 -*-
__author__ = 'yamamoto'
# env: Python2.7

import os
import datetime
import pandas as pd
import numpy as np
import sys

# ################################################
# #  初期入力
# ################################################
#
# # 参加者の名前を入力
Name = 'Hogehoge_hoge'
RecName = ['Recording100', 'Recording101']
# like 'Yamamoto_Hiroki'
# Name = 'Iwata_Chiaki'

################################################
#  処理関数
################################################

def day_input(text=None):
    if text is None:
        if sys.version_info.major > 2:  # 3系かどうかの判定
            return input()
        else:
            return raw_input()  # python2ではraw_input
    else:
        if sys.version_info.major > 2:  # 3系かどうかの判定
            return input(text)
        else:
            return raw_input(text)  # python2ではraw_input


def set_day():
    sday = day_input('Please enter base date (e.g. 2015/06/04, None => today)  ')
    return sday

# 誕生日を入力させる
def get_birthday():
    # 入力待ち(関数を別に作ってPython2系にも対応)
    bday = day_input('Please enter ones birthday (e.g. 2015/06/04)  ')
    return bday


# 年齢の計算（閏日補正含む） ：今何歳何ヶ月なのか？
def count_years(b, s):
    try:
        this_year = b.replace(year=s.year)
    except ValueError:
        b += timedelta(days=1)
        this_year = b.replace(year=s.year)

    age = s.year - b.year
    if base_day < this_year:
        age -= 1

# 何歳”何ヶ月”を計算
    if (s.day - b.day) >= 0:
        year_months = (s.year - b.year) * 12 - age * 12 + (s.month - b.month)
    else:
        year_months = (s.year - b.year) * 12 - age * 12 + (s.month - b.month) - 1  # 誕生日が来るまでは月齢も-1

    return age, year_months


# 月齢の計算
def count_months(b, s):
    if (s.day - b.day) >= 0:
        months = (s.year - b.year) * 12 + (s.month - b.month)
    else:
        months = (s.year - b.year) * 12 + (s.month - b.month) - 1  # 誕生日が来るまでは月齢も-1
    return months


# 月齢および何歳何ヶ月の余り日数（何歳何ヶ月”何日”）
def count_days(b, s):
    if (s.day - b.day) >= 0:
        days = s.day - b.day
    else:
        try:
            if s.month == 1:
                before = s.replace(year=s.year - 1, month=12, day=b.day)
                days = (s - before).days
            else:
                before = s.replace(month=s.month - 1, day=b.day)
                days = (s - before).days
        except ValueError:
            days = s.day
            # 2月は1ヶ月バックするとエラーになる時がある(誕生日が29-31日の時)
            # なのでそうなった場合は、すでに前月の誕生日を迎えたことにする（setされた日が日数とイコールになる）
    return days


################################################
#  メイン処理
################################################

while True:
    DIR_PATH = os.getcwd()
    (root, dirs, files) = next(os.walk(DIR_PATH.rstrip(os.sep)))

    try:
        files.remove('renameFiles_each_5.2.py')
        files.remove('renameFiles_each.py')
    except Exception as e_file:
        print("-----------------------------------")
        print("error occured")
        print(e_file)
        print("-----------------------------------")

    print('root', root)
    print()
    print('dirs', dirs)
    print()
    print('files', files)

    # 参加者データの読み込み
    desktop_path = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop"
    try:
        infants = pd.read_csv(desktop_path + '\\infants.csv')
    except Exception as e_file:
        print("-----------------------------------")
        print("error occured")
        print(e_file)
        print("-----------------------------------")
        break

    try:
        if len(infants.query("Name==@Name")) == 0:
            e_name = "There is no name like " + Name
            print("-----------------------------------")
            print(e_name)
            print("Check participant's name")
            print("-----------------------------------")
            break
        else:
            df = infants.query("Name==@Name")
            print(df)
            print()
    except Exception as e_name:
        print("-----------------------------------")
        print("error occured")
        print(e_name)
        print("-----------------------------------")
        break

    # 調査日程・誕生日から月齢・日齢を計算
    # NameID・調査日程を取得
    idx_nid = df.index[0]  # インデックスを取得
    NameID = df.ix[idx_nid, "NameID"]
    datetime_birth = datetime.datetime.strptime(df.ix[idx_nid, "Birth"], '%Y/%m/%d')

    # 各調査日程フォルダごとにファイルをrename
    # ファイルの更新日時を取得
    GP = {}
    TB = {}
    AD = {}
    EX = {}
    for file in files:
        if (file.endswith('MP4') or file.endswith('mp4')) and (file.startswith('GP') or file.startswith('GO')):
            GP[os.stat(os.path.join(file)).st_mtime] = file
        elif file.endswith('MP4') or file.endswith('mp4'):
            TB[os.stat(os.path.join(file)).st_mtime] = file
        elif file.endswith('MP3') or file.endswith('mp3'):
            AD[os.stat(os.path.join(file)).st_mtime] = file
        elif file.endswith('xlsx'):
            EX[os.stat(os.path.join(file)).st_mtime] = file

    if len(GP) != 0:
        print('GP', GP)
        # 調査日程からfileIDを定義
        datetime_obs = datetime.datetime.fromtimestamp(list(GP.items())[0][0])
        ObsID = NameID + datetime_obs.strftime('%y%m%d')
        i = 0
        for st_mtime, fileName in sorted(list(GP.items())):
            i += 1
            newName = 'GP_' + ObsID + '_' + str(i) + '.mp4'
            os.rename(fileName, newName)

    if len(TB) != 0:
        print('TB', TB)
        # 調査日程からfileIDを定義
        datetime_obs = datetime.datetime.fromtimestamp(list(TB.items())[0][0])
        ObsID = NameID + datetime_obs.strftime('%y%m%d')
        i = 0
        for st_mtime, fileName in sorted(list(TB.items())):
            i += 1
            newName = 'TB_' + ObsID + '_' + str(i) + '.mp4'
            os.rename(fileName, newName)
    else:
        # RecDate = ['NA']
        RecName = np.nan

    if len(AD) != 0:
        # 調査日程からfileIDを定義
        print('AD', AD)
        datetime_obsA = datetime.datetime.fromtimestamp(list(AD.items())[0][0])
        ObsIDa = NameID + datetime_obs.strftime('%y%m%d')
        i = 0
        for st_mtime, fileName in sorted(list(AD.items())):
            i += 1
            newName = 'AD_' + ObsIDa + '_' + str(i) + '.mp3'
            os.rename(fileName, newName)

    if len(EX) != 0 and 'datetime_obs' in locals():
        ObsID = NameID + datetime_obs.strftime('%y%m%d')
        i = 0
        for st_mtime, fileName in sorted(list(EX.items())):
            i += 1
            newName = ObsID + '_' + str(i) + '.xlsx'
            os.rename(fileName, newName)

    if (len(GP) != 0 or len(TB) != 0 or len(AD) != 0 or len(EX) != 0) and 'datetime_obs' in locals():
        RecDate = datetime.datetime.strptime(ObsID[6:12], '%y%m%d').strftime('%Y-%m-%d')

        # 調査日程・誕生日から月齢・日齢を計算
        # 日齢計算
        Age_days = (datetime_obs - datetime_birth).days  # 日齢
        print('age in days:',  Age_days)

        # 月齢計算・余り日数計算
        Months = count_months(datetime_birth, datetime_obs)
        Days = count_days(datetime_birth, datetime_obs)
        print('age in months;', Months)
        print('days:', Days)

        # 調査日程の記録をcsv出力
        df_obs = pd.DataFrame({'Obs': [datetime_obs],
                               'ObsID': [ObsID],
                               'AgeinDays': [Age_days],
                               'Months': [Months],
                               'Days': [Days],
                               'GP': [len(GP)],
                               'TB': [len(TB)],
                               'AD': [len(AD)],
                               'EX': [len(EX)],
                               'RecDate': [RecDate],
                               'RecName': [RecName]},
                               index=[idx_nid])

        df_obs = df_obs[['Obs', 'ObsID', 'AgeinDays', 'Months', 'Days', 'GP', 'TB', 'AD', 'EX', 'RecDate', 'RecName']]
        df_obs = pd.concat([df, df_obs], axis=1)

        if os.path.isfile(desktop_path + '\\Observation.csv'):
            df_old = pd.read_csv(desktop_path + '\\Observation.csv')
            del df_old['SerialID']
            # for idx in df_old.index:
            #     df_old.ix[idx, 'Obs'] = datetime.datetime.strptime(df_old.ix[idx, 'Obs'], '%Y-%m-%d %H:%M:%S')
            df_obs = pd.concat([df_old, df_obs])
            df_obs['Obs'] = pd.to_datetime(df_obs['Obs'])
            # 重複行があれば削除
            # df_obs = df_obs[df_obs['ObsID'].duplicated() == False]
            df_obs = df_obs[df_obs['ObsID'].duplicated(keep='last') == False]
        df_obs = df_obs.sort_values('Obs')  # 観察日順に並び替え
        df_obs.insert(0, 'SerialID', range(1, len(df_obs)+1))  # ObsIDをつける
        df_obs = df_obs.sort_values(['id', 'Obs'])  # id, ObsIDの順にソート

        print df_obs
        df_obs.to_csv(desktop_path + '\\Observation.csv', index=False, na_rep='NA')

        break

        # if root.split("\\")[-1] != fileID:
        #     os.rename(roor, fileID)

        # if os.path.isdir(root):
        #     os.rename(root, fileID)