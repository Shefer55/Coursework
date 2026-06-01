import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3
import nltk
from nltk.corpus import stopwords

# Загрузка данных (файл в формате Excel)
df = pd.read_excel('women-clothing-accessories.3-class.balanced.xlsx')
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(df.head())

# Проверка баланса классов
class_counts = df['sentiment'].value_counts()
print(class_counts)

plt.figure(figsize=(8, 6))
colors = ['#2ecc71', '#f39c12', '#e74c3c']
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
plt.title('Рисунок 4.1. Распределение классов в датасете', fontsize=14)
plt.axis('equal')
plt.show()

def clean_text(text):
    """
    Очистка текста:
    - приведение к нижнему регистру
    - удаление знаков препинания и цифр
    - удаление лишних пробелов
    """
    text = str(text).lower()                     # приведение к нижнему регистру
    text = re.sub(r'[^а-яё\s]', '', text)        # удаление всего, кроме русских букв и пробелов
    text = ' '.join(text.split())                # удаление лишних пробелов
    return text

df['clean_review'] = df['review'].apply(clean_text)

# ВЫВОД ПРИМЕРОВ ДО И ПОСЛЕ ОЧИСТКИ
print("=" * 70)
print("Примеры очистки текста")
print("=" * 70)
for i in range(5):
    print(f"\n{i+1}. ДО ОЧИСТКИ:     {df['review'].iloc[i]}")
    print(f"   ПОСЛЕ ОЧИСТКИ:  {df['clean_review'].iloc[i]}")

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmas)

df['lemmas'] = df['clean_review'].apply(lemmatize_text)

# ВЫВОД ПРИМЕРОВ ДО И ПОСЛЕ ЛЕММАТИЗАЦИИ
print("\n" + "=" * 70)
print("Примеры лемматизации")
print("=" * 70)
for i in range(5):
    print(f"\n{i+1}. ПОСЛЕ ОЧИСТКИ:  {df['clean_review'].iloc[i]}")
    print(f"   ПОСЛЕ ЛЕММАТИЗАЦИИ: {df['lemmas'].iloc[i]}")

# Объединение всех лемматизированных текстов
all_words = ' '.join(df['lemmas']).split()
word_counts = Counter(all_words)

print("Топ-15 самых частых слов:")
for word, count in word_counts.most_common(15):
    print(f"{word}: {count}")

# Построение столбчатого графика
top_words = word_counts.most_common(15)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 7))
plt.bar(words, counts, color='steelblue')
plt.title('Рисунок 4.2. Топ-15 самых частых слов в корпусе отзывов', fontsize=14)
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Создание облака слов
all_text = ' '.join(df['lemmas'])
wc = WordCloud(width=1000, height=500, max_words=100, background_color='white',
               colormap='viridis', random_state=42).generate(all_text)

plt.figure(figsize=(14, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Рисунок 4.3. Облако слов для корпуса отзывов', fontsize=16)
plt.tight_layout()
plt.show()

# Облако слов для позитивных отзывов
pos_text = ' '.join(df[df['sentiment'] == 'positive']['lemmas'])
wc_pos = WordCloud(width=400, height=300, background_color='white', max_words=30,
                   colormap='Greens').generate(pos_text)

# Облако слов для нейтральных отзывов
neut_text = ' '.join(df[df['sentiment'] == 'neautral']['lemmas'])
wc_neut = WordCloud(width=400, height=300, background_color='white', max_words=30,
                    colormap='Blues').generate(neut_text)

# Облако слов для негативных отзывов
neg_text = ' '.join(df[df['sentiment'] == 'negative']['lemmas'])
wc_neg = WordCloud(width=400, height=300, background_color='white', max_words=30,
                   colormap='Reds').generate(neg_text)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Позитивные отзывы', fontsize=14)

axes[1].imshow(wc_neut, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Нейтральные отзывы', fontsize=14)

axes[2].imshow(wc_neg, interpolation='bilinear')
axes[2].axis('off')
axes[2].set_title('Негативные отзывы', fontsize=14)

plt.suptitle('Рисунок 4.4. Облака слов для разных классов тональности', fontsize=16)
plt.tight_layout()
plt.show()

# Скачивание списка стоп-слов
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# Дополнительные стоп-слова
extra_stop_words = {
    'очень', 'всё', 'не', 'очень', 'так', 'такой', 'такое', 'такая', 'такие',
    'ещё', 'уже', 'даже', 'вот', 'вон', 'прямо', 'просто', 'совсем', 'чуть',
    'немного', 'много', 'мало', 'почти', 'слишком', 'также', 'который',
    'которая', 'которое', 'которые', 'этот', 'эта', 'это', 'эти', 'весь',
    'вся', 'всё', 'все', 'свой', 'своя', 'своё', 'свои', 'быть', 'мочь'
}
stop_words.update(extra_stop_words)

def remove_stopwords(text, stop_words):
    """Удаление стоп-слов из текста"""
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['no_stopwords'] = df['lemmas'].apply(lambda x: remove_stopwords(x, stop_words))

# ВЫВОД ПРИМЕРОВ ДО И ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ
print("\n" + "=" * 70)
print("Примеры удаления стоп-слов")
print("=" * 70)
for i in range(5):
    print(f"\n{i+1}. ПОСЛЕ ЛЕММАТИЗАЦИИ:    {df['lemmas'].iloc[i]}")
    print(f"   ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ: {df['no_stopwords'].iloc[i]}")

# Подсчёт частоты слов после удаления стоп-слов
all_words_clean = ' '.join(df['no_stopwords']).split()
word_counts_clean = Counter(all_words_clean)

print("Топ-15 самых частых слов (ПОСЛЕ удаления стоп-слов):")
for word, count in word_counts_clean.most_common(15):
    print(f"{word}: {count}")

# Построение графика
top_words_clean = word_counts_clean.most_common(15)
words_clean, counts_clean = zip(*top_words_clean)

plt.figure(figsize=(12, 7))
plt.bar(words_clean, counts_clean, color='coral')
plt.title('Рисунок 4.5. Топ-15 самых частых слов после удаления стоп-слов', fontsize=14)
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Создание векторайзера TF-IDF
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)
tfidf_matrix = vectorizer.fit_transform(df['no_stopwords'])

# Получение списка всех слов (словаря)
feature_names = vectorizer.get_feature_names_out()
print(f"Размер словаря: {len(feature_names)} слов")
print(f"Первые 20 слов в словаре:\n{feature_names[:20]}")

# ВЫБИРАЕМ ОДИН ТЕКСТ ДЛЯ ПРИМЕРА (например, первый позитивный отзыв)
example_idx = df[df['sentiment'] == 'positive'].index[0]
example_text = df['no_stopwords'].iloc[example_idx]
example_vector = tfidf_matrix[example_idx].toarray()[0]

# СОЗДАЁМ ТАБЛИЦУ 4.7: TF-IDF вектор для выбранного текста
# Получаем ненулевые значения (слова, которые есть в тексте)
non_zero_indices = example_vector.nonzero()[0]
non_zero_words = [feature_names[i] for i in non_zero_indices]
non_zero_values = [example_vector[i] for i in non_zero_indices]

# Создаём DataFrame для таблицы
tfidf_table = pd.DataFrame({
    'Слово': non_zero_words,
    'TF-IDF': [f"{v:.4f}" for v in non_zero_values]
})

print("\n" + "=" * 60)
print(f"Таблица 4.7. Пример TF-IDF вектора для текста:")
print(f"Текст: {example_text[:100]}...")
print("=" * 60)
print(tfidf_table.to_string(index=False))


def search_texts(query, vectorizer, tfidf_matrix, texts_df, text_column='no_stopwords', top_n=3):
    """
    Поиск наиболее похожих текстов по запросу.

    Параметры:
    - query: поисковый запрос
    - vectorizer: обученный векторайзер TF-IDF
    - tfidf_matrix: матрица TF-IDF для корпуса
    - texts_df: DataFrame с текстами
    - text_column: название столбца с текстами для поиска
    - top_n: количество возвращаемых результатов

    Возвращает:
    - список словарей с результатами поиска
    """
    # Очистка и лемматизация запроса
    query_clean = clean_text(query)
    query_lemmatized = lemmatize_text(query_clean)
    query_no_stop = remove_stopwords(query_lemmatized, stop_words)

    # Векторизация запроса
    query_vec = vectorizer.transform([query_no_stop])

    # Вычисление косинусного сходства между запросом и всеми текстами
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Получение индексов top_n наиболее похожих текстов
    top_indices = similarities.argsort()[-top_n:][::-1]

    # Формирование результатов
    results = []
    for idx in top_indices:
        results.append({
            'Текст': texts_df[text_column].iloc[idx],
            'Исходный_текст': texts_df['review'].iloc[idx],
            'Метка': texts_df['sentiment'].iloc[idx],
            'Похожесть': similarities[idx]
        })

    return results


# Демонстрация работы поиска
queries = [
    "хорошее качество товара",
    "плохая доставка долго ждать",
    "не подошел размер вернуть деньги"
]

for query in queries:
    print(f"\n{'=' * 70}")
    print(f"Поисковый запрос: '{query}'")
    print('=' * 70)

    results = search_texts(query, vectorizer, tfidf_matrix, df, top_n=3)

    for i, res in enumerate(results, 1):
        print(f"\n{i}. Похожесть: {res['Похожесть']:.6f}")
        print(f"   Метка: {res['Метка']}")
        print(f"   Текст: {res['Текст'][:200]}...")