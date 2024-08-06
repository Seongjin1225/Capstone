# -*- coding: utf-8 -*-
"""
# 1. 기본 ResNet50 모델을 직접 구현 해 학습
# 2. 전이학습 모델 통한 학습
- ulcer : 각막궤양
- fseques : 각막부골편
- conjunc : 결막염
- nonulcer: 비궤양성각막염
- bleph : 안검염
"""

# 한글 깨짐 방지
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

"""# 필요 라이브러리 호출"""

# 필요 라이브러리 호출
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import pickle
import random

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D, ZeroPadding2D, Add
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

"""# 기본 ResNet50 구현"""

# ResNet50 모델
def conv1_layer(x):
    x = ZeroPadding2D(padding=(3,3))(x)
    x = Conv2D(64,(7,7), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    return x

def conv2_layer(x):
    x = MaxPooling2D((3,3), 2)(x)

    shortcut = x

    for i in range(3):
        if i == 0:
            x = Conv2D(64, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3,3), strides = (1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1,1), strides=(1,1), padding= 'valid')(x)
            shortcut = Conv2D(256, (1,1), strides=(1,1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x
        else:
            x = Conv2D(64, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (3,3), strides = (1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if i == 0:
            x = Conv2D(128, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(512, (1,1), strides=(2,2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x,shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if i == 0:
            x = Conv2D(256, (1,1), strides=(2,2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1,1), strides=(1,1), padding='valid')(x)
            shortcut = Conv2D(1024, (1,1), strides=(2,2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1,1), strides=(1,1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(256, (3,3), strides=(1,1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x

def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if(i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def ResNet50(input_shape):
    inputs = Input(shape=input_shape)
    x = conv1_layer(inputs)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=inputs, outputs=x)
    return model

input_shape = (224,224,3)
model = ResNet50(input_shape)

""" # Raw Data Loading"""

# Train Data
from tqdm import tqdm
import pandas as pd
from pathlib import Path

folder_path = '/content/drive/MyDrive/종합설계/Train/안구/일반'

# 질병 목록 정의
diseases = ['각막궤양', '각막부골편', '결막염', '비궤양성각막염', '안검염']
disease_type = []

# 질병 폴더로부터 '유' 폴더의 경로를 disease_type에 추가
for disease in diseases:
    disease_path = Path(folder_path + '/' + disease + '/유')
    disease_type.append(disease_path)

# '무' 폴더를 처리하기 위한 별도의 리스트 생성 - 모든 '무' 폴더 루트
normal_images = []
for disease in diseases:
    normal_path = Path(folder_path + '/' + disease + '/무')
    normal_images.extend(list(normal_path.iterdir()))

# DataFrame 생성
df = pd.DataFrame()

# '유' 폴더 내 이미지 처리
for types in disease_type:
    for imagepath in tqdm(list(types.iterdir()), desc=str(types)):
        if str(imagepath).endswith('.jpg'):
            df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)], 'disease_type': [disease_type.index(types)]})], ignore_index=True)

# '무' 폴더 내 이미지 처리
for imagepath in tqdm(normal_images, desc="Normal Images"):
    if str(imagepath).endswith('.jpg'):
        df = pd.concat([df, pd.DataFrame({'image': [str(imagepath)], 'disease_type': [5]})], ignore_index=True)

# Valid Data
from tqdm import tqdm
import pandas as pd
from pathlib import Path

folder_path = '/content/drive/MyDrive/종합설계/Valid/안구/일반'

# 질병 목록 정의
diseases = ['각막궤양', '각막부골편', '결막염', '비궤양성각막염', '안검염']
disease_type = []

# 질병 폴더로부터 '유' 폴더의 경로를 disease_type에 추가
for disease in diseases:
    disease_path = Path(folder_path + '/' + disease + '/유')
    disease_type.append(disease_path)

# '무' 폴더를 처리하기 위한 별도의 리스트 생성 - 모든 '무' 폴더 루트
normal_images = []
for disease in diseases:
    normal_path = Path(folder_path + '/' + disease + '/무')
    normal_images.extend(list(normal_path.iterdir()))

# DataFrame 생성
df_val = pd.DataFrame()

# '유' 폴더 내 이미지 처리
for types in disease_type:
    for imagepath in tqdm(list(types.iterdir()), desc=str(types)):
        if str(imagepath).endswith('.jpg'):
            df_val = pd.concat([df_val, pd.DataFrame({'image': [str(imagepath)], 'disease_type': [disease_type.index(types)]})], ignore_index=True)

# '무' 폴더 내 이미지 처리
for imagepath in tqdm(normal_images, desc="Normal Images"):
    if str(imagepath).endswith('.jpg'):
        df_val = pd.concat([df_val, pd.DataFrame({'image': [str(imagepath)], 'disease_type': [5]})], ignore_index=True)

df_val.tail()

"""# 데이터 증강(Data Augumentation) - (1)"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1/255,
                               rotation_range=30,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

valid_gen = ImageDataGenerator(rescale=1/255)

train_generator = train_gen.flow_from_dataframe(df_train,
                                                directory=None,
                                                x_col='image',
                                                y_col='disease_type',
                                                target_size=(224,224),
                                                batch_size=256,
                                                class_mode='categorical')

valid_generator = valid_gen.flow_from_dataframe(df_val,
                                                directory=None,
                                                x_col='image',
                                                y_col='disease_type',
                                                target_size=(224,224),
                                                batch_size=256,
                                                class_mode='categorical')

"""# 모델 학습"""

print("모델 학습 시작.")

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss ='binary_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(f"best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True
                               verbose=1)

history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.samples // batch_size,
                        epochs=100,
                        verbose=1,
                        callbacks=[checkpoint, early_stopping])

with open(f'mode_hist.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print("모델 학습 완료.")

"""# 전이학습 및 파인튜닝 모델 생성
- 각각의 질병에 대해 모델 생성 후 각각의 예측 결과 비교 후
- 최종 예측결과값 출력
"""

# 모델 생성
def make_model(input_shpae):
    base_model = ResNet50(weights='imagenet',input_shape=input_shape,include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (224, 224, 3)
# num_classes = 2

# 각막궤양 모델
model_ulcer = make_model(input_shape)

# 각막부골편 모델
model_fseques = make_model(input_shape)

# 결막염 모델
model_conjunc = make_model(input_shape)

# 비궤양성각막염 모델
model_nonulcer = make_model(input_shape)

# 안검염 모델
model_bleph = make_model(input_shape)

# 모델 예시 확인
model_conjunc.summary()

"""# 데이터 증강(Data Augumentation) - (2)"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 클래스 이름 리스트
classes = ['각막궤양', '각막부골편', '결막염', '비궤양성각막염', '안검염']

# 기본 경로 설정
train_base_path = '/content/drive/MyDrive/종합설계/Train/안구/일반/'
valid_base_path = '/content/drive/MyDrive/종합설계/Valid/안구/일반/'

# ImageDataGenerator 설정
train_gen = ImageDataGenerator(rescale=1/255,
                               rotation_range=30,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

valid_gen = ImageDataGenerator(rescale=1/255)

# 데이터 제너레이터를 저장할 딕셔너리 초기화
train_generators = {}
valid_generators = {}

# 각 클래스에 대해 제너레이터 생성
for class_name in classes:
    train_path = f"{train_base_path}{class_name}/"
    valid_path = f"{valid_base_path}{class_name}/"

    train_generators[class_name] = train_gen.flow_from_directory(directory=train_path,
                                                                 target_size=(224, 224),
                                                                 batch_size=32,
                                                                 class_mode='binary',
                                                                 shuffle=True)

    valid_generators[class_name] = valid_gen.flow_from_directory(directory=valid_path,
                                                                 target_size=(224, 224),
                                                                 batch_size=32,
                                                                 class_mode='binary',
                                                                 shuffle=False)

# 모델 학습
models = {'각막궤양' : model_ulcer,
          '각막부골편' : model_fseques,
          '결막염' : model_conjunc,
          '비궤양성각막염' : model_nonulcer,
          '안검염' : model_bleph}

batch_size = 64
epochs = 100
all_history = {}

for class_name, model in models.items():
    print(f"{class_name} 모델 학습 시작")

    checkpoint = ModelCheckpoint(f"best_model_{class_name}.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='max')

    history = model.fit(train_generators[class_name],
                        steps_per_epoch=train_generators[class_name].samples // batch_size,
                        validation_data=valid_generators[class_name],
                        validation_steps=valid_generators[class_name].samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpoint, early_stopping])

    all_history[class_name] = history.history
    with open(f'history_{class_name}.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print(f"{class_name} 모델 학습 완료.")

"""# 학습 결과 시각화"""

histories = {}
for class_name in models.keys():
    with open(f'history_{class_name}.pkl', 'rb') as f:
        histories[class_name] = pickle.load(f)

# 시각화 함수 정의
def plot_history(history, class_name):
    plt.figure(figsize=(12, 5))
    plt.rc('font', family='NanumBarunGothic')

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{class_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{class_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# 각 모델의 학습 기록 시각화
for class_name, history in histories.items():
    plot_history(history, class_name)

"""# 결과 예측 테스트"""

# 각 질병별로 훈련된 모델 로드
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

model_ulcer = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_각막궤양.h5')
model_fseques = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_각막부골편.h5')
model_conjunc = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_결막염.h5')
model_nonulcer = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_비궤양성각막염.h5')
model_bleph = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_안검염.h5')

def load_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 정규화 (0-1 사이 값)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 예시 이미지
image = load_image('/content/drive/MyDrive/종합설계/test.jpg')

preds_1 = model_ulcer.predict(image)
preds_2 = model_fseques.predict(image)
preds_3 = model_conjunc.predict(image)
preds_4 = model_nonulcer.predict(image)
preds_5 = model_bleph.predict(image)

# 예측결과 출력
preds_list = [preds_1, preds_2, preds_3, preds_4, preds_5]

diseases = ["각막궤양", "각막부골편", "결막염", "비궤양성각막염", "안검염"]

class_preds = []
probabilities = []

for preds in preds_list:
    class_pred = 1 if preds[0] >= 0.5 else 0
    class_preds.append(class_pred)
    probabilities.append(preds[0])

if all(prob <= 0.5 for prob in probabilities):
    print("정상")
else:
    max_prob_index = probabilities.index(max(probabilities))
    max_disease = diseases[max_prob_index]
    max_class_pred = class_preds[max_prob_index]

    print(f"최대 확률을 가진 질병: {max_disease}, 클래스: {max_class_pred}")

"""# Flask 연결"""

!pip install flask-ngrok
!pip install pyngrok

!ngrok authtoken 'my-autotoken'

# !pip install flask pyngrok
from flask import Flask, request, jsonify
from pyngrok import ngrok
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 모델 로드
model_ulcer = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_각막궤양.h5')
model_fseques = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_각막부골편.h5')
model_conjunc = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_결막염.h5')
model_nonulcer = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_비궤양성각막염.h5')
model_bleph = load_model('/content/drive/MyDrive/종설 결과/epoch_100/best_model_안검염.h5')

# 이미지 로드 및 전처리 함수
def load_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 예측 및 결과 반환 엔드포인트
@app.route("/predict", methods=["POST"])
def predict():
    image_path = request.json["image_path"]
    image = load_image(image_path)

    preds_1 = model_ulcer.predict(image)
    preds_2 = model_fseques.predict(image)
    preds_3 = model_conjunc.predict(image)
    preds_4 = model_nonulcer.predict(image)
    preds_5 = model_bleph.predict(image)

    preds_list = [preds_1, preds_2, preds_3, preds_4, preds_5]
    diseases = ["각막궤양", "각막부골편", "결막염", "비궤양성각막염", "안검염"]

    class_preds = []
    probabilities = []

    for preds in preds_list:
        class_pred = 1 if preds[0] >= 0.5 else 0
        class_preds.append(class_pred)
        probabilities.append(preds[0][0])

    if all(prob <= 0.5 for prob in probabilities):
        result = "정상"
    else:
        max_prob_index = probabilities.index(max(probabilities))
        max_disease = diseases[max_prob_index]
        result = f"최대 확률을 가진 질병: {max_disease}, 확률: {probabilities[max_prob_index]}"

    return jsonify({"result": result})

if __name__ == '__main__':
    ngrok.set_auth_token("2hLxclwFEZRhtFvgSmC2nzdRcQA_56FoWMpat7baQc2wPmtYT")  # 여기에 ngrok 인증 토큰을 입력하세요.
    ngrok_tunnel = ngrok.connect(5000)
    print('Public URL:', ngrok_tunnel.public_url)
    app.run()

!pip install flask pyngrok

from flask import Flask
from pyngrok import ngrok

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

if __name__ == '__main__':
  ngrok.set_auth_token("2hLxclwFEZRhtFvgSmC2nzdRcQA_56FoWMpat7baQc2wPmtYT")
  ngrok_tunnel = ngrok.connect(5000)
  print('Public URL:', ngrok_tunnel.public_url)
  app.run()