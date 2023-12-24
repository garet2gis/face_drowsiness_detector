import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from models.cnn.model import Net1

model = Net1()
model.load_state_dict(torch.load('models/cnn/net1.pth'))  # Укажите путь к вашей модели
model.eval()

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cap = cv2.VideoCapture(0)

while True:
    # Захват кадра с вебкамеры
    ret, frame = cap.read()

    # Преобразование кадра к 228x228 и нормализация
    pil_img = transform(frame)
    img_tensor = pil_img.unsqueeze(0)

    # Предсказание с использованием модели
    with torch.no_grad():
        prediction = model(img_tensor)

    # Получение метки класса (бодр/устал)
    _, predicted_class = torch.max(prediction, dim=1)

    label = 'Awake' if predicted_class[0].item() == 0 else 'Tired'

    # Вывод результата на кадр
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
