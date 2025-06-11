import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os


# Добавляем путь к директории с b0_модель.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Загрузка модели
@st.cache_resource
def load_model():
    class EyeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.classes = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'S']

        for param in self.base_model.blocks[-4:].parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)
        
    try:
        # Инициализация модели с теми же параметрами
        config = Config()
        model = EyeModel().to('cpu')

        # Загрузка весов
        checkpoint = torch.load(
            os.path.join("best_model.pth"),
            map_location='cpu',
            weights_only=False
        )

        # Загрузка state_dict с обработкой возможных ключей
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

        model.eval()
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return None


# Функция для предобработки изображения
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# Функция для предсказания
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return model.classes[predicted.item()], probabilities.numpy()


# Интерфейс Streamlit
def main():
    st.title("Классификация изображений глаз")
    st.write("Загрузите изображение для определения патологии")

    uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_container_width=True)

        model = load_model()
        if model is None:
            return

        input_tensor = preprocess_image(image)

        if st.button("Определить патологию"):
            class_name, probs = predict(input_tensor, model)

            st.subheader("Результат:")
            st.success(f"Предсказанный класс: {class_name}")

            st.bar_chart({k: v for k, v in zip(model.classes, probs)})

            class_descriptions = {
                'N': 'Норма',
                'D': 'Диабетическая ретинопатия',
                'G': 'Глаукома',
                'C': 'Катаракта',
                'A': 'Возрастная макулодистрофия',
                'H': 'Гипертоническая ретинопатия',
                'M': 'Миопия',
                'S': 'Рубец на макуле'
            }

            st.write("**Описание классов:**")
            for cls, desc in class_descriptions.items():
                st.write(f"- **{cls}**: {desc}")


if __name__ == "__main__":
    main()
