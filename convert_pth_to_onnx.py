import torch
import torch.nn as nn
import onnx
import os

class QNetwork(nn.Module):
    def __init__(self, input_size=20, output_size=4):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        return self.fc(x)

# 1. Загружаем веса из PyTorch
model = QNetwork()
model.load_state_dict(torch.load("snake_dqn_model.pth", map_location=torch.device('cpu')))
model.eval()

# 2. Создаем входной тензор
dummy_input = torch.randn(1, 20)

print("Шаг 1: Экспорт из PyTorch во временные файлы...")
torch.onnx.export(
    model,
    dummy_input,
    "temp_model.onnx",
    export_params=True,
    opset_version=18,
    input_names=['input'],
    output_names=['output']
)

print("Шаг 2: Принудительное объединение графа и весов в один файл...")
# Загружаем модель, используя встроенную библиотеку onnx
# Она автоматически подтянет веса из .data файла, если он создался
onnx_model = onnx.load("temp_model.onnx")

# Сохраняем обратно, но БЕЗ указания внешних форматов.
# Функция onnx.save по умолчанию зашивает всё внутрь одного .onnx файла, если размер < 2ГБ.
onnx.save(onnx_model, "snake_model.onnx")

# Чистим за собой временные файлы
if os.path.exists("temp_model.onnx"):
    os.remove("temp_model.onnx")
if os.path.exists("temp_model.onnx.data"):
    os.remove("temp_model.onnx.data")

print("Успешно! Теперь у вас есть ровно ОДИН монолитный файл: snake_model.onnx")