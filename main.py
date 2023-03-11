import os
# импортируем библиотеку pathlib, а также функцию Path для работы с директориями
from pathlib import Path

import matplotlib.pyplot as plt
# для упорядочивания файлов в директории
import natsort
import splitfolders
import tensorflow as tf
# библиотеки для работы с изображениями
from PIL import Image
from tensorflow import keras

input_folder = "Faces"

# -- Импорт для подготовки данных: --
# модуль для предварительной обработки изображений

# Класс ImageDataGenerator - для генерации новых изображений на основе имеющихся
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -- Импорт для построения модели: --
# импорт слоев нейросети
from tensorflow.keras import layers
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import RMSprop

# Получим и отсортируем список с названиями фото с женскими лицами
woman_filenames = os.listdir("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Female")
woman_filenames = natsort.natsorted(woman_filenames)

# Получим и отсортируем список с названиями фото с мужскими лицами
men_filenames = os.listdir("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Male")
men_filenames = natsort.natsorted(men_filenames)

# -- Выведем часть фотографий (с 5 по 15) на экран:
# 1. создаем график(фигуру) для вывода всех фото
pic_box = plt.figure(figsize=(14, 12))
for i, image_name in enumerate(woman_filenames[5:15]):
    # 2. считываем текущее изображение
    image = plt.imread(str(Path("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Female", image_name)))
    # 3. создаем "подграфик" для вывода текущего изображения в заданной позиции
    ax = pic_box.add_subplot(3, 5, i + 1)
    # 4. в качестве названия графика определяем имя фотографии и число каналов
    ax.set_title(str(image_name) + '\n Каналов = ' + str(image.shape[2]))
    # 5. выводим изображение на экран
    plt.imshow(image)
    # 6. отключаем вывод осей графика
    plt.axis('off')
plt.show()

for img in woman_filenames:
    im = Image.open(Path("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Female", img))
    # Если расширение файла ".png" и формат файла "PNG":
    if img[-3:].lower() == 'png' and im.format == 'PNG':
        # если режим изображения не RGBA (без альфа-канала):
        if im.mode != 'RGBA':
            # конвертируем фото в RGBA и сохраняем в той же директории под тем же именем
            im.convert("RGBA").save(Path("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Female", img))
            # при желании, можно вывести имена файлов, которые были переформатированы.
            print(img)
for img in os.listdir("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Male"):
    im = Image.open(Path("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Male", img))
    # Если расширение файла ".png" и формат файла "PNG":
    if img[-3:].lower() == 'png' and im.format == 'PNG':
        # если режим изображения не RGBA (без альфа-канала):
        if im.mode != 'RGBA':
            # конвертируем фото в RGBA и сохраняем в той же директории под тем же именем
            im.convert("RGBA").save(Path("C:/Users/gabdu/PycharmProjects/Нейронка/Faces/Male", img))
            # при желании, можно вывести имена файлов, которые были переформатированы.
            print(img)

splitfolders.ratio("Faces", 'faces_splited', ratio=(0.8, 0.15, 0.05), seed=18, group_prefix=None)

# определим параметры нормализации данных
train = ImageDataGenerator(rescale=1 / 255)
val = ImageDataGenerator(rescale=1 / 255)

# сгенерируем нормализованные данные
train_data = train.flow_from_directory('faces_splited/train', target_size=(299, 299),
                                       class_mode='binary', batch_size=3, shuffle=True)
val_data = val.flow_from_directory('faces_splited/val', target_size=(299, 299),
                                   class_mode='binary', batch_size=3, shuffle=True)

# Определяем параметры аугментации
data_augmentation = keras.Sequential(
    [
        # Отражение по горизонтали
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(299, 299, 3)),
        # Вращение на рандомное значение до 0.05
        layers.experimental.preprocessing.RandomRotation(0.05),
        # Меняем контрастность изображений
        layers.experimental.preprocessing.RandomContrast(0.23),
        # Изменяем размер
        layers.experimental.preprocessing.RandomZoom(0.2)
    ]
)

model = Sequential([
    # добавим аугментацию данных
    data_augmentation,
    layers.Conv2D(16, (3, 3), activation='selu', input_shape=(299, 299, 3)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='selu'),
    layers.MaxPool2D(2, 2),
    layers.Dropout(0.05),

    layers.Conv2D(64, (3, 3), activation='selu'),
    layers.MaxPool2D(2, 2),
    layers.Dropout(0.1),
    layers.Conv2D(128, (2, 2), activation='selu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(256, (2, 2), activation='selu'),
    layers.MaxPool2D(2, 2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(500, activation='selu'),

    layers.Dense(1, activation='sigmoid')
])

# Файл для сохранения модели с лучшими параметрами
checkpoint_filepath = 'best_model.h5'
# Компиляция модели
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.00024),
              # optimizer=tf.keras.optimizers.Adam(learning_rate=0.000244),
              metrics=['binary_accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True)

# Тренировка модели
history = model.fit(train_data, batch_size=500, verbose=1, epochs=35,
                    validation_data=val_data,
                    callbacks=[model_checkpoint_callback])
