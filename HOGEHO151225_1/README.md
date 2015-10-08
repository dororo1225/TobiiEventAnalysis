TobiiEventAnalysys_4.1
====

## Overview
Tobii Glasses2のRaw Dataを用いて解析をおこなうためのPython & Rスクリプト

## Prerequest
- Tobii Glasses2のRaw Dataをtsvファイルとして出力しておく('Data Export.csv').
- 処理したいデータの(Analysys Softwareにおける) Recording Number, Participant Name, Recording Dateをメモしておく.
- Dropboxに "AnnotationAssistant" フォルダがあることを確認しておく.
- Desktopに 'infants.csv' があることを確認しておく.
- 処理したいデータのmp4ファイルとtsvファイルをTobiiEventAnalysys_4.1.pyのあるディレクトリに移動させておく.

## Usage
#### 1. TobiiEventAnalysys_4.1.pyを実行する.
[Process]
 - Accelerometer dataのSmoothing, 水平面に対する頭部角度の計算
 - 画像回転 ・ Face Detection


 [Output]
  - Gaze_HODEHOGE.csv : Raw Dataのcsv出力
  - Event_HOGEHOGE.csv : Event生起時のRaw Data
  - Origin_HOGEHOGE.csv : Face DetectionするEventID, FrameID, RotateIDのリスト
  - Face_HOGEHOGE.csv : Face Detectionのデータを含めたdataframe
  - "output" フォルダ : Face DetectionでAnnotationが付加された画像

#### 2. preprocess1.Rmdを実行する.
[Process]
 - 'Event_HOGEHOGE.csv' の読み込み, Event間隔からSessionを定義
 - 'Face_HOGEHOGE.csv' の読み込み, SessionごとにAnnotation Sizeのmean ・ SDを計算
 - 各Sessionについて, Height sizeが外れ値となるAnnotationを除去
 - 複数のAnnotationがある画像について, すべてのAnnotationを除去
 - AnnotationがひとつもないSessionがあるかどうか確認    
 [AnnotationがひとつもないSessionがある場合]
  - 処理後の 'Face_HOGEHOGE.csv' 由来のデータをcsv出力( 'unprocData_HOGEHOGE,csv' )
  - AnnotationがひとつもないSessionの基本情報をcsv出力( 'Annot_HOGEHOGE.csv' )


 [Output]
  - unprocData_HODEHOGE.csv : 前処理後のRaw Dataのcsv出力
  - Annot_HOGEHOGE.csv : AnnotationがひとつもないSessionの基本情報
  - processedData_HOGEHOGE.csv :  前処理後のRaw Dataのcsv出力(AnnotationのないSessionがない場合)
  - Session_HOGEHOGE.csv    : 各Sessionの基本情報(AnnotationのないSessionがない場合)

#### 3. Copy.pyを実行する.
[Process]
 - 'Annot_HOGEHOGE.csv' の読み込み
 - AnnotationがひとつもないSessionの画像をコピーして, "AnnotationAssistant"内のフォルダに移動する


 [Output]
 - "AnnotationAssistant/FaceClipper/static/images"

#### 4. "AnnotationAssistant"フォルダ → "DBForamat"フォルダのDBFormat.pyを実行する.
[Process]
 - Copy.pyで移動した画像のpath等の情報を, DBとしてDrobox上に保存する

 [Output]
 - 'Dropbox/AnnotationAssistant/HOGEHOGE.db'

#### 5. "AnnotationAssistant"フォルダ → "FaceClipper"フォルダのStart.batを実行する.
[Process]
 - AnnotationがないSessionの画像について, 手動でAnnotationをつける
 - Dropbox上の 'HOGEHOGE.db' のデータを更新する

#### 6. "AnnotationAssistant"フォルダ → "DBOutput"フォルダのDBOutput.pyを実行する.
[Process]
 - 'HOGEHOGE.db'のデータをcsv出力する


 [Output]
 - 'AnnotationAssistant/DBOutput/AnnotationData.csv'

#### 7. preprocess2.Rmdを実行する.
[Process]
 - 'AnnotationAssistant/DBOutput/AnnotationData.csv'を読み込む
 - 'unprocData_HOGEHOGE.csv' を読み込み, AnnotationなしSessionの行を, 手動コードのデータで更新する
 - 処理後の 'unprocDara_HOGEHOGE.csv' 由来のデータをcsv出力( 'processedData_HOGEHOGE,csv' )
 - 各Sessionの基本情報をcsv出力( 'Session_HOGEHOGE.csv' )

 [Output]
 - processedData_HOGEHOGE.csv :  前処理後のRaw Dataのcsv出力
 - Session_HOGEHOGE.csv    : 各Sessionの基本情報

## Author

[dororo1225](https://github.com/dororo1225)
