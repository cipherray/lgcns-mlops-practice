# 적절한 위치에 맞는 수준으로 로그 출력되도록 코드 작성

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    LOG_FILEPATH,
    PREDICTION_PATH,
)
from src.common.logger import handle_exception, set_logger

# 로그를 정해진 로그 경로에 logs.log로 저장하도록 설정
logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception

sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")

if __name__ == "__main__":
    DATE = datetime.now().strftime("%Y%m%d")

    # 테스트 데이터 불러오기
    test = pd.read_csv(os.path.join(DATA_PATH, "house_rent_test.csv"))

    # joblib.load() 로 베스트 모델 불러오기
    model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))

    X = test.drop(["id", "rent"], axis=1, inplace=False)
    id_ = test["id"].to_numpy()

    # 테스트 데이터에 대한 피처 데이터 저장
    # 전체 파이프라인에서 preprocessor 파이프라인만 가져와서 transform() 사용
    model["preprocessor"].transform(X=X).to_csv(
        os.path.join(DATA_PATH, "storage", "house_rent_test_features.csv"),
        index=False,
    )

    # 예측값을 저장한 데이터프레임
    pred_df = pd.DataFrame({"user": id_, "rent": np.expm1(model.predict(X))})
    logger.info(f"Batch prediction for {len(pred_df)} users is created...")

    save_path = os.path.join(PREDICTION_PATH, f"{DATE}_rent_prediction.csv")
    pred_df.to_csv(save_path, index=False)
