from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from joblib import load
from typing import List

# GPU 사용 가능 여부 확인
device = torch.device("mps")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# LSTMModel 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size).to(device)
        self.linear = nn.Linear(hidden_layer_size, output_size).to(device)
        
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        
        return predictions

scaler = load("./minmax_scaler.pkl")

model = LSTMModel().to(device)
model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
model.eval()  # 평가 모드로 설정

app = FastAPI()

# Pydantic을 사용한 입력 데이터 모델
class InputData(BaseModel):
    sequence: List[List[float]] = Field(..., example=[[0.1, 0.2], [0.3, 0.4]])  # 예시 추가

    # sequence의 크기 검증을 위한 validator 추가
    @classmethod
    def check_sequence_size(cls, v, values, **kwargs):
        if len(v) != 10 or any(len(row) != 2 for row in v):
            raise ValueError('Sequence must be of size (10, 2)')
        return v

@app.post("/predict/")
async def predict(data: InputData):
    # 입력 데이터를 스케일링
    scaled_input = scaler.transform(data.sequence)
    # 입력 데이터를 텐서로 변환 및 GPU로 이동
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0).to(device)  # 배치 차원 추가
    with torch.no_grad():  # 추론 모드
        predictions = model(input_tensor)
    # 예측 결과를 역스케일링
    predictions_numpy = predictions.cpu().numpy()  # GPU에서 CPU로 이동
    original_predictions = scaler.inverse_transform(predictions_numpy)  # 역스케일링
    import numpy as np
    return {"predictions": original_predictions.tolist(), "predictions_numpy": predictions_numpy.tolist(), "label": scaler.transform(np.array([[40.002938,116.325151]]))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
