import os
import sys
import datetime

import pytz
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Any

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from collections import deque

from setting import *

tf.random.set_seed(314)
random.seed(314)
np.random.seed(314)


def check_folder():
    if not os.path.isdir("models"):
        os.mkdir("models")

    if not os.path.isdir("logs"):
        os.mkdir("logs")


def read_csv_file(data_path, feature_columns):
    df = pd.read_csv(data_path, header=None, index_col=0)
    # 이때 첫번쨰 열(%Y%m%d) 과 두번째 열(%H%M) 을 합쳐서 index로 지정함.
    df.index = pd.to_datetime(df.index.astype(str) + df[1].astype(str), format="%Y%m%d%H%M")
    # 첫번째 열과 두번째 열을 합쳐서 index 로 만들고 두번째 열은 제거한다.
    df = df.drop(1, axis=1)  
    # 컬럼명을 지정합니다.
    df.columns = ["open", "high", "low", "close", "volume"]
    # 중복된 index 제거. 중복된 index가 있으면 데이터가 이상하게 저장되어 있을 수 있다.
    df = df[~df.index.duplicated(keep="first")]
    # feature_columns 에 지정된 컬럼만 사용한다. ex) ["open", "high", "low", "close"]
    df = df[feature_columns]  
    return df


def load_data(df, n_steps=N_STEPS, scale=True, lookup_step=LOOKUP_STEP,
              split_by_date=True, test_size=TEST_SIZE, feature_columns=["open", "high", "low", "close"]):
    # lookup_step : 1분봉은 60개, 5분봉은 300개, 10분봉은 600개, 15분봉은 900개, 30분봉은 1800개, 1시간봉은 3600개
    # 원본 result 딕셔너리에 복사
    result = {"df": df.copy()}  
    # 만약 "date" 컬럼이 없으면 추가하고
    if "date" not in df.columns:
        df["date"] = df.index       # "date" 컬럼에 index 를 넣음.

    # 스케일링은 데이터를 0~1 사이로 변환함
    if scale:
        column_scaler = {}
        for column in feature_columns:
            scaler = MinMaxScaler()  # MinMaxScaler 는 데이터를 0~1 사이로 변환
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))  # 2차원 배열로 변환
            column_scaler[column] = scaler  # 스케일러를 column_scaler 딕셔너리에 저장

        result["column_scaler"] = column_scaler  # 스케일러를 result 딕셔너리에 저장
        print("스케일링을 완료했습니다.")

    # Try using .loc[row_indexer,col_indexer] = value instead 에러를 피하기 위해 스케일링을 먼저 수행한다.
    print("데이터셋을 생성합니다.")
    # 미래 컬럼을 생성합니다.
    df["future"] = df["close"].shift(-lookup_step)  # 다음날 종가를 future 컬럼에 저장한다.

    # 마지막 lookup_step 열에는 NaN 값이 있으므로 삭제하기전에 미리 저장한다.
    last_sequence = np.array(df[df.columns[:-1]].tail(lookup_step))
    # 마지막 lookup_step 열을 삭제한다.
    df.dropna(inplace=True)
    print("데이터셋을 생성했습니다.")

    print("시퀀스 데이터를 생성합니다..")
    # 시퀀스 데이터를 생성하는 1번 방법.
    sequence_data = []  # 이 시퀀스 데이터는 X와 y로 분할됩니다.
    sequences = deque(maxlen=n_steps)  # n_steps 만큼의 데이터를 저장할 수 있는 큐를 생성한다.
    # 데이터프레임의 모든 행을 반복합니다.
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):  # df[feature_columns + ["date"]].values는 시퀀스 데이터를 만드는 데 사용된다.
        sequences.append(entry)  # 큐에 데이터를 추가
        if len(sequences) == n_steps:  # 큐에 n_steps 만큼의 데이터가 쌓이면
            sequence_data.append([np.array(sequences), target])  # 시퀀스 데이터에 추가

    # 마지막 시퀀스 데이터를 2차원으로 생성
    last_sequence = np.array([s[:len(feature_columns)] for s in sequences] + [last_sequence], ndmin=2, dtype=object)
    # 결과에 추가
    result["last_sequence"] = last_sequence
    # X와 y를 구성합니다.
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)

    print("X와 y를 구성했습니다")

    # 분할하지 않으려면 아래 주석을 해제합니다.
    # result["X_train"] = X  # 이렇게 했는데 왜 훈련 데이터가 없는지 모르겠다. 이 부분을 수정해야함.
    # result["y_train"] = y
    # return result
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=False)

    result["y_train"] = result["y_train"].astype(np.int)
    result["y_test"] = result["y_test"].astype(np.int)

    print("데이터를 분할했습니다.")
    # 테스트 세트 날짜 목록 가져 오기
    dates = result["X_test"][:, -1, -1].flatten() # 마지막 열의 마지막 열을 가져온다
    # 원래 데이터 프레임에서 테스트 기능 검색
    result["test_df"] = result["df"].loc[dates] # 테스트 날짜를 기준으로 데이터 프레임을 분할
    print("test_df :", result["test_df"].shape)
    # 테스트 세트에서 중복된 날짜 제거
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # 훈련 / 테스트 세트에서 날짜 제거하고 float32로 변환
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result


def create_model(model_path):
    print("모델을 생성합니다.")
    model: Any = Sequential()
    model.add(LSTM(UNITS, return_sequences=True, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(UNITS, return_sequences=True, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(UNITS, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='linear'))  # softmax 는 0~1 사이의 값으로 출력한다. linear 는 음수와 양수를 모두 출력함.
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_model(model_path):
    try:
        if os.path.isfile(model_path):
            print("모델을 로드합니다: ", model_path)
            model: Any = tf.keras.models.load_model(model_path)
        else:
            model = create_model(model_path)
    except Exception as e:
        print("모델을 로드하는 동안 문제가 발생했습니다. 모델을 삭제하고 새로 생성합니다.")
        print(e)
        os.remove(model_path)
        model = create_model(model_path)
    return model


def plot_graph(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f'true_close_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'close_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
    plt.savefig('./graphs/prediction.png')


def predict(model, data):
    last_sequence = data["last_sequence"][-N_STEPS:]
    last_sequence = last_sequence.astype(np.float32)
    last_sequence = np.array([last_sequence])

    prediction = model.predict(last_sequence)   # 예측을 가져온다. (0에서 1로 스케일링됨)
    if SCALE:
        predicted_price = data["column_scaler"]["close"].inverse_transform(prediction)[0][0]    # 역 스케일링
    else:
        predicted_price = prediction[0][0]  # 역 스케일링을 하지 않는 경우 예측을 그대로 사용

    return predicted_price


def get_final_df(model, data):
    # 예측된 가격이 현재 가격보다 높으면 1, 낮으면 0, 같으면 2, 현재 가격에서 차익을 계산
    buy_profit = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # 예측된 가격이 현재 가격보다 낮으면 1, 높으면 0, 같으면 2, 현재 가격에서 차익을 계산
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"] # 테스트 데이터
    y_test = data["y_test"] # 테스트 데이터의 정답
    # 예측을 수행하고 가격을 얻습니다.
    y_pred = model.predict(X_test)

    if SCALE:   # 스케일 복구
        y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0))) # 정답 데이터의 스케일을 원래대로 되돌린다
        y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))                         # 예측 데이터의 스케일을 원래대로 되돌린다

    # 테스트 세트에 대한 최종 데이터 프레임을 가져온다
    test_df = data["test_df"]
    print("data[test_df] :", data["test_df"].shape)
    print("test_df :", test_df.shape)
    #  test_df에 예측된 가격 추가
    test_df[f"close_{LOOKUP_STEP}"] = y_pred[:, 0]

    # test_df에 실제 가격 추가
    test_df[f"true_close_{LOOKUP_STEP}"] = y_test
    # test_df를 날짜별로 정렬한다
    test_df.sort_index(inplace=True)
    final_df = test_df
    # test_df에 매수 수익률 추가
    final_df["buy_profit"] = list(map(buy_profit,
                                      final_df["close"],
                                      final_df[f"close_{LOOKUP_STEP}"],
                                      final_df[f"true_close_{LOOKUP_STEP}"])
                                  # 마지막 시퀀스에 대한 수익률은 0으로 추가
                                  )
    # test_df에 매도 수익률 추가
    final_df["sell_profit"] = list(map(sell_profit,
                                       final_df["close"],
                                       final_df[f"close_{LOOKUP_STEP}"],
                                       final_df[f"true_close_{LOOKUP_STEP}"])
                                   # 마지막 시퀀스에 대한 수익률은 0으로 추가
                                   )
    return final_df


def short_summary(code):
    summary = [
        {
            "Ticker": code,
            f"Future price after": f"{LOOKUP_STEP} day",
            f"Predicted price for {tomorrow}": f"{future_price:.2f}$",
            "Mean absolute error": mean_absolute_error,
            "Accuracy score": accuracy_score,
            "Total buy profit": total_buy_profit,
            "Total sell profit": total_sell_profit,
            "Total profit": total_profit,
            "Profit per trade": profit_per_trade,
            "Generated": current_time
        }
    ]
    """save data to json file"""
    with open("data.json", "w") as outfile:
        json.dump(summary, outfile, indent=4, sort_keys=False)
    return summary


if __name__ == '__main__':
    # 그래픽카드 사용 가능하면 True, 아니면 False 출력
    print("그래픽카드 사용 가능: ", tf.test.is_gpu_available())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
        except RuntimeError as e:
            print(e)

    check_folder()
    # 만약 파라미터로 --test 가 들어오면 테스트 모드로 실행한다.
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        학습모드 = False
    else:
        학습모드 = True

    종목코드 = "A000020"
    종목풀코드 = "KR7000020008"
    타임프레임 = "m"
    기간 = "1"
    기본데이터경로 = os.path.join(STATIC_DATA_PATH, 종목풀코드, f"{종목코드}_{타임프레임}{기간}.csv")
    print("데이터를 읽어옵니다. 파일명 : ", 기본데이터경로)
    학습데이터 = read_csv_file(기본데이터경로, feature_columns=FEATURE_COLUMNS)
    # 데이터 전처리
    학습준비데이터 = load_data(학습데이터, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                        lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                        feature_columns=FEATURE_COLUMNS)

    모델이름 = f"{종목코드}-{CELL.__name__}-{scale_str}-{split_by_date_str}-{LOSS}-{OPTIMIZER}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    # ex) 000020-LSTM-minmax-False-mse-Adam-seq-50-step-3-layers-3-units-128
    모델파일경로 = os.path.join("models", 모델이름) + ".h5"
    모델 = get_model(모델파일경로)
    if 학습모드:
        print("학습을 시작합니다.")
        # 몇 가지 tensorflow 콜백
        print("모델을 저장합니다.")
        checkpointer = ModelCheckpoint(모델파일경로, save_weights_only=False, save_best_only=True, verbose=1)
        # ModelCheckpoint 함수는 모델을 저장합니다. save_weights_only=True 는 가중치만 저장하고, save_best_only=True 는 가장 좋은 모델만 저장한다.
        # 모델 전체를 저장하려면 save_weights_only=False 으로
        print("모델을 학습시킵니다.")
        tensorboard = TensorBoard(log_dir=os.path.join("logs", 모델이름))
        # 모델을 훈련하고 새로운 최적 모델을 볼 때마다 가중치 저장
        y_train = np.eye(2)[학습준비데이터["y_train"]]
        y_test = np.eye(2)[학습준비데이터["y_test"]]

        history = 모델.fit(학습준비데이터["X_train"], 학습준비데이터["y_train"],
                         batch_size=BATCH_SIZE,  # fit 함수에서 batch_size 는 한 번에 학습시킬 데이터의 양
                         epochs=1,  # fit 함수에서 epochs 는 전체 데이터를 몇 번 학습시키는지
                         validation_data=(학습준비데이터["X_test"], 학습준비데이터["y_test"]),  # 검증 데이터 지정
                         callbacks=[checkpointer, tensorboard],  # 콜백을 지정합니다.
                         verbose=1)  # 학습 진행 상황 출력, 갱신하지 않고 새로운 출력을 하게 하려면 verbose=0 으로
        print("학습이 끝났습니다.")
    else:
        print("검증을 시작합니다.")
        loss, mae = 모델.evaluate(학습준비데이터["X_test"], 학습준비데이터["y_test"], verbose=1)
        # 모델.evaluate 함수는 모델 평가, 학습 데이터를 넣어서 학습 데이터에 대한 정확도 확인 가능.
        print(f"모델 손실: {loss}, 평균절대오차: {mae}")

        if SCALE:
            mean_absolute_error = 학습준비데이터["column_scaler"]["close"].inverse_transform([[mae]])[0][0]
        else:
            mean_absolute_error = mae

        final_df = get_final_df(모델, 학습준비데이터)
        future_price = predict(모델, 학습준비데이터)
        # 정확도는 양수 수익을 세는 것으로 계산
        accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(
            final_df)

        total_buy_profit = final_df["buy_profit"].sum()
        total_sell_profit = final_df["sell_profit"].sum()
        total_profit = total_buy_profit + total_sell_profit
        profit_per_trade = total_profit / len(final_df)

        print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
        print(f"{LOSS} loss:", loss)
        print("Mean Absolute Error:", mean_absolute_error)
        print("Accuracy score:", accuracy_score)
        print("Total buy profit:", total_buy_profit)
        print("Total sell profit:", total_sell_profit)
        print("Total profit:", total_profit)
        print("Profit per trade:", profit_per_trade)
        
        plot_graph(final_df)
        print(final_df.tail(10))
        
        # 파일명 ex) 2020-01-01_2020-01-31_1day_10lookback_10epochs_64batchsize_0.001lr_0.2dropout_0.2validation_split
        results_folder = "results"
        결과파일명 = os.path.join(results_folder, 모델이름 + ".csv")
        final_df.to_csv(결과파일명)

        # 서버에서 로그를 남기기 위해 현재 시간을 얻어옴.
        KL = pytz.timezone("Asia/Seoul")
        current_time = str(datetime.datetime.now(KL))

        # 1일 뒤의 가격을 예측합니다.
        tomorrow = datetime.date.today() + datetime.timedelta(days=int(LOOKUP_STEP))
        short_summary(종목코드)

