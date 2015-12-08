# coding:utf-8
__author__ = 'yamamoto'

import os
import shutil

obs_id = os.getcwd().split('\\')[-1]
pic_name = obs_id.split('TB_')[1] + '.png'

##############################
# Data要約 周辺
##############################

files = os.listdir(os.getcwd())  # ディレクトリ内のファイル名を取得
for file in files:
    if file.startswith('Gaze_') or file.startswith('Event_') or file.startswith('Face_') or file.startswith('Summary_'):
          os.remove(file)

if os.path.isfile(pic_name):
    os.remove(pic_name)

#  FaceClipper\static\imagesを消去する
if os.path.isdir('output'):
    shutil.rmtree('output')

##############################
# Annotation Assistant 周辺
##############################

os.chdir('AnnotationAssistant')

#  FaceClipper\static\imagesを消去する
if os.path.isdir('FaceClipper\\static\\images'):
    shutil.rmtree('FaceClipper\\static\\images')

if os.path.isdir('FaceClipper\\static\\' + obs_id):
    shutil.rmtree('FaceClipper\\static\\' + obs_id)

if os.path.isfile('FaceClipper\\static\\' + obs_id + '.txt'):
    os.remove('FaceClipper\\static\\' + obs_id + '.txt')

# DBOutput\\bg.txtを消去する
if os.path.isfile('DBOutput\\bg.txt'):
    os.remove('DBOutput\\bg.txt')

# DBOutput\\info.datを消去する
if os.path.isfile('DBOutput\\info.dat'):
    os.remove('DBOutput\\info.dat')

# DBOutput\\info.txtを消去する
if os.path.isfile('DBOutput\\info.txt'):
    os.remove('DBOutput\\info.txt')

# DBOutput\\TeacherData.csvを消去する
if os.path.isfile('DBOutput\\TeacherData.csv'):
    os.remove('DBOutput\\TeacherData.csv')

# DBOutput\\AnnotationData.csvを消去する
if os.path.isfile('DBOutput\\progress.csv'):
    os.remove('DBOutput\\progress.csv')

# DBOutput\\AnnotationData.csvを消去する
if os.path.isfile('DBOutput\\AnnotationData.csv'):
    os.remove('DBOutput\\AnnotationData.csv')


# Dropbox内のDBファイルの初期化
obs_id = os.getcwd().split('\\')[-1]
db_name = obs_id + '.db'
data_path = os.getenv("HOMEDRIVE") + \
                    os.getenv("HOMEPATH") +  \
                    "\\Dropbox\\AnnotationAssistant\\" + db_name
print data_path

# if os.path.isfile(data_path):
#     os.remove(data_path)

#  Dropbox\\test_db内のsamples.dbを消去する
#  DBファイルの場所としてDropboxのpathを取得
data_path = os.getenv("HOMEDRIVE") + \
            os.getenv("HOMEPATH") +  \
            "\\Dropbox\\teach_db\\samples.db"

if os.path.isfile(data_path):
    os.remove(data_path)