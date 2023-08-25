import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    bhk: int
    size: int
    floor: str
    area_type: str
    city: str
    furnishing_status: str
    tenant_preferred: str
    bathroom: int
    point_of_contact: str


# 학습 코드에서 저장한 베스트 모델을 가져올 것 (house_rent:latest)
bento_model = bentoml.sklearn.get("house_rent:latest")

# 모델 러너 생성
model_runner = bento_model.to_runner()

# 모델 러너로 서비스 생성
# "rent_house_regressor"라는 이름으로 서비스를 띄우기
svc = bentoml.Service("rent_house_regressor", runners=[model_runner])


@svc.api(
    # Features 클래스를 JSON으로 받아오고 Numpy NDArray를 반환하도록 데코레이터 작성
    input=JSON(pydantic_model=Features),
    # bentoml.io 에서 지정된 타입을 골라서 적음
    output=NumpyNdarray,
)
async def predict(input_data: Features) -> npt.NDArray:
    input_df = pd.DataFrame([input_data.dict()])
    # 비동기 async 는 await 과 함께 다님
    log_pred = await model_runner.predict.async_run(input_df)
    return np.expm1(log_pred)
