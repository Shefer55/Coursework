# =============================================================================
# КУРСОВОЙ ПРОЕКТ - АНАЛИЗ ДАННЫХ TITANIC
# Глава 1: Первичный анализ табличных данных (ПОЛНАЯ ВЕРСИЯ)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. ЗАГРУЗКА НЕОБХОДИМЫХ БИБЛИОТЕК
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# 2. ЗАГРУЗКА ДАННЫХ
# -----------------------------------------------------------------------------

df = pd.read_csv('Titanic-Dataset.csv', delimiter=';')

print("ПЕРВЫЕ 5 СТРОК ДАТАСЕТА:")
print(df.head())
print("\n" + "="*60 + "\n")

print("ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
print(df.info())
print("\n" + "="*60 + "\n")

print("ОСНОВНЫЕ СТАТИСТИКИ ДЛЯ ЧИСЛОВЫХ ПРИЗНАКОВ:")
print(df.describe())
print("\n" + "="*60 + "\n")

print(f"Всего пассажиров: {len(df)}")
print(f"Выжило: {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)")
print(f"Погибло: {len(df) - df['Survived'].sum()} ({(1-df['Survived'].mean())*100:.1f}%)")

# -----------------------------------------------------------------------------
# 3. ДИАГРАММЫ РАСПРЕДЕЛЕНИЯ (Matplotlib)
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

df['Survived'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['red', 'green'])
axes[0, 0].set_title('Распределение выживших (1) и погибших (0)')
axes[0, 0].set_xlabel('Статус (0=погиб, 1=выжил)')
axes[0, 0].set_ylabel('Количество пассажиров')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Погиб (0)', 'Выжил (1)'])

df['Age'].dropna().hist(bins=30, ax=axes[0, 1], edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].set_title('Распределение возраста пассажиров')
axes[0, 1].set_xlabel('Возраст (лет)')
axes[0, 1].set_ylabel('Количество пассажиров')

df['Parch'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Количество родителей/детей на борту')
axes[1, 0].set_xlabel('Количество родственников')
axes[1, 0].set_ylabel('Количество пассажиров')

df['Pclass'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='skyblue')
axes[1, 1].set_title('Распределение по классам билета')
axes[1, 1].set_xlabel('Класс (1=первый, 2=второй, 3=третий)')
axes[1, 1].set_ylabel('Количество пассажиров')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4. ВИЗУАЛИЗАЦИЯ СРЕДСТВАМИ SEABORN
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[0], color='steelblue')
axes[0].set_title('Выживаемость в зависимости от класса билета')
axes[0].set_xlabel('Класс билета')
axes[0].set_ylabel('Доля выживших')

sns.barplot(x='Sex', y='Survived', data=df, ax=axes[1], hue='Sex', palette=['coral', 'lightgreen'], legend=False)
axes[1].set_title('Выживаемость в зависимости от пола')
axes[1].set_xlabel('Пол')
axes[1].set_ylabel('Доля выживших')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df, hue='Pclass', palette='Set2', legend=False)
plt.title('Распределение возраста по классам билета')
plt.xlabel('Класс билета')
plt.ylabel('Возраст (лет)')
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df[['Survived', 'Pclass', 'Age', 'Parch']].dropna()
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляция между числовыми признаками')
plt.show()

# -----------------------------------------------------------------------------
# 5. ИНТЕРАКТИВНЫЕ ГРАФИКИ (Plotly)
# -----------------------------------------------------------------------------

fig = px.histogram(df, x='Age', nbins=30, title='Интерактивное распределение возраста')
fig.show()

fig2 = px.box(df, x='Pclass', y='Age', color='Pclass',
              title='Распределение возраста по классам билета')
fig2.show()

# -----------------------------------------------------------------------------
# 6. АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
print("="*60)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    'Количество пропусков': missing,
    'Доля пропусков (%)': missing_pct
})
print(missing_table[missing_table['Количество пропусков'] > 0])

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Тепловая карта пропущенных значений (жёлтое = пропуск)')
plt.show()

# -----------------------------------------------------------------------------
# 7. ЗАПОЛНЕНИЕ ПРОПУСКОВ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ЗАПОЛНЕНИЕ ПРОПУСКОВ:")
print("="*60)

before_age = df['Age'].isnull().sum()
before_embarked = df['Embarked'].isnull().sum()

for pclass in [1, 2, 3]:
    for sex in ['male', 'female']:
        median_age = df[(df['Pclass'] == pclass) & (df['Sex'] == sex)]['Age'].median()
        mask = (df['Pclass'] == pclass) & (df['Sex'] == sex) & (df['Age'].isnull())
        df.loc[mask, 'Age'] = median_age

most_common_port = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(most_common_port)

print(f"Пропусков в Age до: {before_age}, после: {df['Age'].isnull().sum()}")
print(f"Пропусков в Embarked до: {before_embarked}, после: {df['Embarked'].isnull().sum()}")

# -----------------------------------------------------------------------------
# 8. ПРОВЕРКА ДУБЛИКАТОВ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ПРОВЕРКА ДУБЛИКАТОВ:")
print("="*60)

duplicates_count = df.duplicated().sum()
print(f"Количество полных дубликатов строк: {duplicates_count}")

if duplicates_count > 0:
    df = df.drop_duplicates()
    print("Дубликаты удалены")
else:
    print("Дубликатов не найдено")

# -----------------------------------------------------------------------------
# 9. ПОИСК ВЫБРОСОВ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("АНАЛИЗ ВЫБРОСОВ:")
print("="*60)

Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print(f"Нижняя граница нормы: {lower_bound:.1f} лет")
print(f"Верхняя граница нормы: {upper_bound:.1f} лет")
print(f"Количество выбросов по возрасту: {len(outliers)}")

if len(outliers) > 0:
    print("\nПримеры выбросов (пожилые пассажиры):")
    print(outliers[['Age', 'Sex', 'Pclass', 'Survived']].head(10))

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Age'], color='lightblue')
plt.title('Ящик с усами для возраста (точки за усами = выбросы)')
plt.xlabel('Возраст (лет)')
plt.show()

# -----------------------------------------------------------------------------
# 10. УСЛОВНАЯ ФИЛЬТРАЦИЯ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("УСЛОВНАЯ ФИЛЬТРАЦИЯ ДАННЫХ:")
print("="*60)

filter1 = df[(df['Sex'] == 'male') & (df['Age'] >= 18) & (df['Pclass'] == 1)]
print(f"Фильтр 1 (мужчины 18+, 1 класс): {len(filter1)} пассажиров")

filter2 = df[(df['Sex'] == 'female') & (df['Survived'] == 1)]
print(f"Фильтр 2 (выжившие женщины): {len(filter2)} пассажиров")

filter3 = df[df['Parch'] > 0]
print(f"Фильтр 3 (путешествуют с семьёй): {len(filter3)} пассажиров")

print("\nПримеры семейных пассажиров:")
print(filter3[['Name', 'Age', 'Parch', 'Survived']].head())

# -----------------------------------------------------------------------------
# 11. ДОБАВЛЕНИЕ ШУМА
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ДОБАВЛЕНИЕ ШУМА:")
print("="*60)

df_noisy = df.copy()

np.random.seed(42)
noise = np.random.normal(0, 3, size=len(df_noisy))
df_noisy['Age_noisy'] = df_noisy['Age'] + noise

def add_inversion_noise(value, prob=0.05):
    if np.random.random() < prob:
        return np.random.randint(0, 5)
    return value

df_noisy['Parch_noisy'] = df_noisy['Parch'].apply(add_inversion_noise)

print("Созданы новые столбцы с шумом:")
print("  - Age_noisy: возраст + случайный шум (σ=3)")
print("  - Parch_noisy: инверсионный шум с вероятностью 5%")

print("\nСравнение оригинального и зашумлённого возраста (первые 5 строк):")
comparison = pd.DataFrame({
    'Оригинальный возраст': df['Age'].head(),
    'Зашумлённый возраст': df_noisy['Age_noisy'].head()
})
print(comparison)

# -----------------------------------------------------------------------------
# 12. ПРЕОБРАЗОВАНИЕ ВОЗРАСТА В КАТЕГОРИИ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ПРЕОБРАЗОВАНИЕ ВОЗРАСТА В КАТЕГОРИИ:")
print("="*60)

def age_to_group(age):
    if age <= 12:
        return 'Ребёнок (0-12)'
    elif age <= 18:
        return 'Подросток (13-18)'
    elif age <= 40:
        return 'Взрослый (19-40)'
    elif age <= 60:
        return 'Средний возраст (41-60)'
    else:
        return 'Пожилой (61+)'

df['Age_group'] = df['Age'].apply(age_to_group)

age_group_counts = df['Age_group'].value_counts()
print("Распределение по возрастным группам:")
for group, count in age_group_counts.items():
    print(f"  {group}: {count} чел. ({count/len(df)*100:.1f}%)")

survival_by_age = df.groupby('Age_group')['Survived'].mean().sort_values(ascending=False)
print("\nВыживаемость по возрастным группам:")
for group, survival in survival_by_age.items():
    print(f"  {group}: {survival*100:.1f}%")

plt.figure(figsize=(10, 5))
age_order = ['Ребёнок (0-12)', 'Подросток (13-18)', 'Взрослый (19-40)', 'Средний возраст (41-60)', 'Пожилой (61+)']
sns.barplot(x='Age_group', y='Survived', data=df, order=age_order, hue='Age_group', legend=False, palette='viridis')
plt.title('Выживаемость по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Доля выживших')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 13. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
print("="*60)

df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})
df_with_dummies = pd.get_dummies(df, columns=['Embarked'], prefix='Порт')

print("Созданы новые числовые признаки:")
print("  - Sex_num: male=0, female=1")
print("  - Порт_C, Порт_Q, Порт_S: бинарные признаки для порта посадки")

# -----------------------------------------------------------------------------
# 14. ГРУППИРОВКА ДАННЫХ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ГРУППИРОВКА ДАННЫХ:")
print("="*60)

grouped = df.groupby(['Sex', 'Pclass']).agg({
    'Survived': ['mean', 'count'],
    'Age': 'mean'
}).round(2)

grouped.columns = ['Выживаемость (%)', 'Количество', 'Ср. возраст']
grouped['Выживаемость (%)'] = grouped['Выживаемость (%)'] * 100

print("Статистика по группам (пол + класс билета):")
print(grouped)

plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, palette=['coral', 'lightgreen'])
plt.title('Выживаемость по классу билета и полу')
plt.xlabel('Класс билета')
plt.ylabel('Доля выживших')
plt.legend(title='Пол')
plt.show()

# -----------------------------------------------------------------------------
# 15. ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ - СЕМЕЙНЫЙ СТАТУС:")
print("="*60)

df['Family_status'] = df['Parch'].apply(lambda x: 'С семьёй' if x > 0 else 'Один(а)')

family_stats = df.groupby('Family_status')['Survived'].agg(['count', 'mean'])
family_stats['mean'] = family_stats['mean'] * 100
family_stats.columns = ['Количество', 'Выживаемость (%)']

print("Анализ по семейному статусу:")
print(family_stats)

plt.figure(figsize=(8, 5))
sns.barplot(x='Family_status', y='Survived', data=df, hue='Family_status', legend=False, palette='Set2')
plt.title('Выживаемость: одинокие vs путешествующие с семьёй')
plt.xlabel('Семейный статус')
plt.ylabel('Доля выживших')
plt.show()

# -----------------------------------------------------------------------------
# 16. ОЦЕНКА ИЗМЕНЕНИЙ ПОСЛЕ ФИЛЬТРАЦИИ (РИСУНОК 1.12)
# -----------------------------------------------------------------------------

print("\n" + "="*60)
print("ОЦЕНКА ИЗМЕНЕНИЙ ПОСЛЕ ФИЛЬТРАЦИИ:")
print("="*60)

filtered_df = df[(df['Sex'] == 'male') & (df['Age'] >= 18) & (df['Pclass'] == 1)]

print(f"Средний возраст ДО фильтрации: {df['Age'].mean():.1f} лет")
print(f"Средний возраст ПОСЛЕ фильтрации: {filtered_df['Age'].mean():.1f} лет")
print(f"Разница: {filtered_df['Age'].mean() - df['Age'].mean():.1f} лет")

# Рисунок 1.12 - Сравнение распределения возраста
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df['Age'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {df["Age"].mean():.1f}')
axes[0].set_title('Распределение возраста (ВСЕ пассажиры)')
axes[0].set_xlabel('Возраст (лет)')
axes[0].set_ylabel('Количество пассажиров')
axes[0].legend()

axes[1].hist(filtered_df['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(filtered_df['Age'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Среднее: {filtered_df["Age"].mean():.1f}')
axes[1].set_title('Распределение возраста (мужчины 18+, 1 класс)')
axes[1].set_xlabel('Возраст (лет)')
axes[1].set_ylabel('Количество пассажиров')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nСтатистика ДО фильтрации:")
print(f"  - Медиана возраста: {df['Age'].median():.1f} лет")
print(f"  - Стандартное отклонение: {df['Age'].std():.1f}")

print("\nСтатистика ПОСЛЕ фильтрации:")
print(f"  - Медиана возраста: {filtered_df['Age'].median():.1f} лет")
print(f"  - Стандартное отклонение: {filtered_df['Age'].std():.1f}")

# -----------------------------------------------------------------------------
# 17. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ
# -----------------------------------------------------------------------------

df.to_csv('Titanic-Dataset-Processed.csv', index=False, sep=';')
print("\nОбработанный датасет сохранён в файл 'Titanic-Dataset-Processed.csv'")

# =============================================================================
# ВЫВОДЫ
# =============================================================================

print("\n" + "="*60)
print("ВЫВОДЫ ПО РЕЗУЛЬТАТАМ АНАЛИЗА:")
print("="*60)

print("""
1. РАЗМЕР И СТРУКТУРА:
   - 891 пассажир, 7 исходных признаков
   - Целевая переменная: Survived (выжил/погиб)

2. ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:
   - Пропуски: Age (177 шт.), Embarked (2 шт.) - УСПЕШНО ЗАПОЛНЕНЫ
   - Дисбаланс классов: выжило 38%, погибло 62%
   - Выбросы: 21 пожилой пассажир - ОСТАВЛЕНЫ

3. КЛЮЧЕВЫЕ ЗАКОНОМЕРНОСТИ:
   - Женщины выживают чаще мужчин (74% vs 19%)
   - Пассажиры 1 класса выживают чаще 3 класса (63% vs 24%)
   - Дети выживают чаще пожилых (60% vs 22%)

4. ПРИГОДНОСТЬ ДАННЫХ: ПРИГОДНЫ для решения задачи классификации
""")