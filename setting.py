import os
import time
from tensorflow.keras.layers import LSTM

STATIC_DATA_PATH = "F:/BetProj/data-static"      # 정적 데이터가 저장된 폴더입니다.
READY_DATA_PATH = "F:/BetProj/data-ready"        # 학습전 데이터를 저장할 폴더입니다.
RESULTS_DATA_PATH = "F:/BetProj/data-results"    # 학습후 데이터를 저장할 폴더입니다.

# 시퀀스 길이 또는 윈도우 크기
N_STEPS = 48    # 5분봉의 경우 48개는 4시간
# '조회 단계' 라고 하는데, 1은 다음날을 의미한다.
LOOKUP_STEP = 6 # 5분봉의 경우 6개는 30분
TEST_SIZE = 0.2 # 테스트 데이터셋의 비율
# 특성 열을 스케일링할지 여부와 출력 가격도 스케일링할지 여부
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# 데이터셋을 섞을지 여부
SHUFFLE = False
shuffle_str = f"sh-{int(SHUFFLE)}"
# 날짜별로 훈련/테스트 세트를 나눌지 랜덤으로 나눌지 여부
SPLIT_BY_DATE = True
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# 테스트 비율 크기, 0.2는 20%를 의미한다.

# 사용할 특성
FEATURE_COLUMNS = ["open", "high", "low", "close"]
# 현재 날짜
date_now = time.strftime("%Y-%m-%d")

### 모델 파라미터

N_LAYERS = 2
# LSTM 셀, LSTM은 RNN의 한 종류로, RNN은 순환 신경망으로 시퀀스 데이터를 처리하는데 사용된다.
CELL = LSTM
# 256 LSTM 뉴런, LSTM은 RNN의 한 종류로, RNN은 순환 신경망으로 시퀀스 데이터를 처리하는데 사용된다.
UNITS = 256
# 40% 드롭아웃, 드롭아웃은 뉴런을 랜덤하게 꺼서 과적합을 방지하는 기법이다.
DROPOUT = 0.4
# 양방향 RNN을 사용할지 여부, 양방향 RNN은 시퀀스의 앞뒤를 모두 보는 것이다.
BIDIRECTIONAL = False

### 훈련 파라미터

# 평균 절대 오차 손실
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 240    # 한번에 처리할 데이터의 양
EPOCHS = 500

# # Amazon stock market
# ticker = "AMZN"
# ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# # model name to save, making it as unique as possible based on parameters
# model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
# {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
# # ex) 2020-12-31_AMZN-sh-1-sc-1-sbd-0-huber_loss-adam-LSTM-seq-50-step-1-layers-2-units-256

