import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image
import os
from pathlib import Path

# === НАСТРОЙКА ===
# Укажите полный путь к вашей папке Dataset
# ВАШ ПУТЬ:
DATA_PATH = Path(r"C:\Users\ksysh\Desktop\Dataset")


# Структура папок должна быть следующей:
# C:\Users\ksysh\Desktop\Dataset\
#     cat/    (200 изображений кошек)
#     dog/    (200 изображений собак)

def load_images_from_folder(folder_path, label, max_count=200):
    """Загружает изображения из папки, не более max_count."""
    images = []
    labels = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Проверяем, существует ли папка
    if not folder_path.exists():
        print(f"ОШИБКА: Папка не найдена: {folder_path}")
        return images, labels

    files = [f for f in os.listdir(folder_path)
             if Path(f).suffix.lower() in valid_extensions]

    if len(files) == 0:
        print(f"ВНИМАНИЕ: В папке {folder_path} не найдено изображений")
        return images, labels

    # Берём первые max_count изображений
    selected_files = files[:max_count]

    for filename in selected_files:
        img_path = folder_path / filename
        try:
            img = Image.open(img_path)
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка загрузки {filename}: {e}")
    return images, labels


# Загрузка данных
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)
print(f"Путь к датасету: {DATA_PATH}")
print(f"Путь к папке с кошками: {DATA_PATH / 'cat'}")
print(f"Путь к папке с собаками: {DATA_PATH / 'dog'}")
print()

# Проверка существования папок
if not DATA_PATH.exists():
    print(f"ОШИБКА: Папка {DATA_PATH} не существует!")
    print("Проверьте правильность пути.")
    exit(1)

print("Загрузка изображений кошек...")
cat_images, cat_labels = load_images_from_folder(DATA_PATH / "cat", 0, 200)
print(f"Загружено кошек: {len(cat_images)}")

print("Загрузка изображений собак...")
dog_images, dog_labels = load_images_from_folder(DATA_PATH / "dog", 1, 200)
print(f"Загружено собак: {len(dog_images)}")

# Проверка, что данные загрузились
if len(cat_images) == 0 and len(dog_images) == 0:
    print("\nОШИБКА: Не удалось загрузить изображения!")
    print("Убедитесь, что внутри папки Dataset есть подпапки 'cat' и 'dog'")
    print("и в них находятся изображения в форматах .jpg, .jpeg или .png")
    exit(1)

# Объединение
all_images = cat_images + dog_images
all_labels = cat_labels + dog_labels

print(f"\nВсего изображений: {len(all_images)}")
print(f"  Кошки: {all_labels.count(0)}")
print(f"  Собаки: {all_labels.count(1)}")
print()

# ============================================================
# 1. Анализ количества и баланса классов
# ============================================================
print("=" * 60)
print("1. АНАЛИЗ КОЛИЧЕСТВА И БАЛАНСА КЛАССОВ")
print("=" * 60)

label_counts = Counter(all_labels)
class_names = {0: "cat", 1: "dog"}

print("\nРаспределение классов в выборке:")
for label, count in label_counts.items():
    print(f"  {class_names[label]}: {count} изображений ({count / len(all_labels) * 100:.1f}%)")

# Визуализация распределения
plt.figure(figsize=(8, 5))
plt.bar([class_names[l] for l in label_counts.keys()],
        label_counts.values(),
        color=['orange', 'brown'])
plt.title('Распределение классов в выборке (200 кошек + 200 собак)', fontsize=14)
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество изображений', fontsize=12)
for i, (label, count) in enumerate(label_counts.items()):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# ============================================================
# 2. Примеры типичных изображений
# ============================================================
print("\n" + "=" * 60)
print("2. ПРИМЕРЫ ТИПИЧНЫХ ИЗОБРАЖЕНИЙ")
print("=" * 60)


def show_examples(images, labels, label_value, num=3):
    """Возвращает первые num изображений указанного класса."""
    indices = [i for i, lbl in enumerate(labels) if lbl == label_value][:num]
    return [images[i] for i in indices]


cat_examples = show_examples(all_images, all_labels, 0, 3)
dog_examples = show_examples(all_images, all_labels, 1, 3)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Примеры изображений из выборки (200 кошек + 200 собак)', fontsize=16)

for i, img in enumerate(cat_examples):
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Кошка {i + 1}', fontsize=12)
    axes[0, i].axis('off')

for i, img in enumerate(dog_examples):
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Собака {i + 1}', fontsize=12)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('examples_images.png', dpi=150)
plt.show()

# ============================================================
# 3. Оценка качества изображений (размеры, разрешение)
# ============================================================
print("\n" + "=" * 60)
print("3. ОЦЕНКА КАЧЕСТВА ИЗОБРАЖЕНИЙ")
print("=" * 60)

widths = []
heights = []
aspect_ratios = []

for img in all_images:
    w, h = img.size
    widths.append(w)
    heights.append(h)
    aspect_ratios.append(w / h)

print(f"\nАнализ размеров {len(widths)} изображений:")
print(f"  Ширина: мин = {min(widths)}px, макс = {max(widths)}px, среднее = {np.mean(widths):.1f}px")
print(f"  Высота: мин = {min(heights)}px, макс = {max(heights)}px, среднее = {np.mean(heights):.1f}px")
print(
    f"  Соотношение сторон: мин = {min(aspect_ratios):.2f}, макс = {max(aspect_ratios):.2f}, среднее = {np.mean(aspect_ratios):.2f}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(widths, bins=20, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Распределение ширины изображений', fontsize=12)
axes[0].set_xlabel('Ширина (пиксели)')
axes[0].set_ylabel('Частота')

axes[1].hist(heights, bins=20, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('Распределение высоты изображений', fontsize=12)
axes[1].set_xlabel('Высота (пиксели)')
axes[1].set_ylabel('Частота')

plt.tight_layout()
plt.savefig('size_distribution.png', dpi=150)
plt.show()

# Маленькие изображения
small_images_count = sum(1 for w in widths if w < 200)
print(f"\nИзображений с шириной менее 200px: {small_images_count} ({small_images_count / len(widths) * 100:.1f}%)")

# ============================================================
# 4. Анализ цветовых режимов
# ============================================================
print("\n" + "=" * 60)
print("4. АНАЛИЗ ЦВЕТОВЫХ РЕЖИМОВ")
print("=" * 60)

modes = Counter()
for img in all_images:
    modes[img.mode] += 1

print("\nЦветовые режимы изображений:")
for mode, count in modes.items():
    print(f"  {mode}: {count} изображений ({count / len(all_images) * 100:.1f}%)")

# ============================================================
# 5. Информация о разметке
# ============================================================
print("\n" + "=" * 60)
print("5. ИНФОРМАЦИЯ О РАЗМЕТКЕ")
print("=" * 60)

print("\nРазметка выполнена на уровне папок:")
print(f"  Папка 'cat' -> класс {class_names[0]}")
print(f"  Папка 'dog' -> класс {class_names[1]}")
print(f"  Всего классов: 2")
print(f"  Размер выборки: {len(all_images)}")