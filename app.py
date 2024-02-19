from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import torch
import torch.nn as nn
from typing import List

# LSTMModel 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=100, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_time_step_out = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step_out)
        
        return predictions

# 모델 인스턴스 생성 및 로드
model = LSTMModel()
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()  # 평가 모드로 설정

app = FastAPI()

# Pydantic을 사용한 입력 데이터 모델
class InputData(BaseModel):
    sequence: List[List[float]]

    # sequence의 크기 검증을 위한 validator 추가
    @field_validator('sequence')
    def check_sequence_size(cls, value):
        if len(value) != 10 or any(len(row) != 2 for row in value):
            raise ValueError('Sequence must be of size (10, 2)')
        return value

@app.post("/predict/")
async def predict(data: InputData):
    # 입력 데이터를 텐서로 변환
    input_tensor = torch.tensor(data.sequence, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():  # 추론 모드
        predictions = model(input_tensor)
    return {"predictions": predictions.numpy().tolist()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)