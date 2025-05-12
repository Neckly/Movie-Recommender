from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, explode, split, count, desc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session():
    #Создание Spark сессии
    try:
        return SparkSession.builder \
            .appName("MovieRecommender") \
            .getOrCreate()
    except Exception as e:
        logger.error(f"Ошибка при создании Spark сессии: {str(e)}")
        raise

def load_data(spark, data_path):
    #Загрузка данных из CSV файлов
    try:
        ratings_path = os.path.join(data_path, "ratings.csv")
        movies_path = os.path.join(data_path, "movies.csv")
        
        ratings = spark.read.csv(ratings_path, header=True, inferSchema=True) \
            .withColumnRenamed("userId", "userID") \
            .withColumnRenamed("movieId", "movieID") \
            .select("userID", "movieID", "rating", "timestamp") \
            .filter(col("rating") > 0)
            
        movies = spark.read.csv(movies_path, header=True, inferSchema=True) \
            .withColumnRenamed("movieId", "movieID") \
            .select("movieID", "title", "genres")
            
        # Удаление пользователей с менее чем 5 оценками
        user_counts = ratings.groupBy("userID").count()
        valid_users = user_counts.filter(col("count") >= 5).select("userID")
        ratings = ratings.join(valid_users, on="userID")
        
        logger.info(f"Загружено {ratings.count()} оценок и {movies.count()} фильмов")
        return ratings, movies
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        raise

def train_model(ratings):
    #Обучение модели ALS
    try:
        als = ALS(
            maxIter=10,
            regParam=0.3,
            rank=10,
            userCol="userID",
            itemCol="movieID",
            ratingCol="rating",
            coldStartStrategy="drop"
        )
        model = als.fit(ratings)
        logger.info("Модель успешно обучена")
        return model
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

def evaluate_model(model, test_data):
    #Оценка качества модели
    try:
        predictions = model.transform(test_data)
        predictions = predictions.withColumn(
            "prediction",
            when(predictions.prediction < 0.5, 0.5)
            .when(predictions.prediction > 5.0, 5.0)
            .otherwise(predictions.prediction)
        )
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        logger.info(f"RMSE на тестовой выборке: {rmse}")
        return rmse
    except Exception as e:
        logger.error(f"Ошибка при оценке модели: {str(e)}")
        raise

def plot_ratings_distribution(ratings_pd):
    #Визуализация распределения оценок
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=ratings_pd, x="rating", bins=20)
        plt.title("Распределение оценок фильмов")
        plt.xlabel("Оценка")
        plt.ylabel("Количество")
        plt.savefig("ratings_distribution.png")
        plt.close()
        logger.info("График распределения оценок сохранен")
    except Exception as e:
        logger.error(f"Ошибка при создании графика: {str(e)}")

def analyze_user_genres(ratings_df, movies_df, user_id):
    #Анализ жанров для конкретного пользователя
    try:
        # Получаем фильмы, которые оценил пользователь
        user_ratings = ratings_df.filter(col("userID") == user_id)
        
        # Присоединяем информацию о жанрах
        user_movies = user_ratings.join(movies_df, "movieID")
        
        # Разбиваем жанры на отдельные строки
        genres_exploded = user_movies.select(
            "movieID",
            "rating",
            explode(split(col("genres"), "\\|")).alias("genre")
        )
        
        # Считаем количество оценок по жанрам
        genre_counts = genres_exploded.groupBy("genre") \
            .agg(
                count("*").alias("count"),
                (count("*") * 100.0 / genres_exploded.count()).alias("percentage")
            ) \
            .orderBy(desc("count"))
            
        return genre_counts
    except Exception as e:
        logger.error(f"Ошибка при анализе жанров пользователя: {str(e)}")
        raise

def plot_genre_distribution(genre_counts, user_id):
    #Визуализация распределения жанров
    try:
        # Преобразуем в pandas для визуализации
        genre_pd = genre_counts.toPandas()
        
        # Создаем фигуру
        plt.figure(figsize=(12, 6))
        
        # Создаем bar plot
        sns.barplot(data=genre_pd, x="genre", y="percentage")
        
        # Настраиваем внешний вид
        plt.title(f"Распределение жанров для пользователя {user_id}")
        plt.xlabel("Жанр")
        plt.ylabel("Процент фильмов")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Сохраняем график
        plt.savefig(f"user_{user_id}_genres.png")
        plt.close()
        
        logger.info(f"График распределения жанров сохранен для пользователя {user_id}")
    except Exception as e:
        logger.error(f"Ошибка при создании графика жанров: {str(e)}")

def get_recommendations(model, ratings_df, movies_df, user_id, n_recommendations=20):
    #Получение рекомендаций для пользователя
    try:
        user_subset = ratings_df.filter(col("userID") == user_id).select(["userID", "movieID"])
        user_recs = model.recommendForUserSubset(user_subset, n_recommendations)
        
        print(f"\nРекомендации для пользователя {user_id}:")
        print("\n{:<60} {:<30} ".format("Название фильма", "Жанры"))
        print("-" * 90)
        
        # Собираем все жанры из рекомендаций для анализа
        recommended_genres = []
        
        for row in user_recs.collect():
            for movie_id, rating in row.recommendations:
                rating = min(rating, 5.0)  # Ограничение до 5.0
                movie_info = movies_df.filter(col("movieID") == movie_id).first()
                if movie_info:
                    title = movie_info.title
                    genres = movie_info.genres
                    recommended_genres.extend(genres.split("|"))
                    
                    # Обрезаем длинные строки
                    if len(title) > 57:
                        title = title[:54] + "..."
                    if len(genres) > 27:
                        genres = genres[:24] + "..."
                        
                    print("{:<60} {:<30} ".format(title, genres))
        
        # Анализируем и визуализируем жанры рекомендаций
        if recommended_genres:
            genre_counts = pd.Series(recommended_genres).value_counts()
            plt.figure(figsize=(12, 6))
            genre_counts.plot(kind='bar')
            plt.title(f"Жанры в рекомендациях для пользователя {user_id}")
            plt.xlabel("Жанр")
            plt.ylabel("Количество фильмов")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"user_{user_id}_recommended_genres.png")
            plt.close()
                    
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise

def main():
    try:
        # Создание Spark сессии
        spark = create_spark_session()
        
        # Загрузка данных
        ratings, movies = load_data(spark, "ml-100k")
        
        # Разделение данных на обучающую и тестовую выборки
        train, test = ratings.randomSplit([0.8, 0.2], seed=42)
        
        # Обучение модели
        model = train_model(train)
        
        # Оценка модели
        rmse = evaluate_model(model, test)
        
        # Визуализация распределения оценок
        ratings_pd = ratings.toPandas()
        plot_ratings_distribution(ratings_pd)
        
        # Пример получения рекомендаций для пользователя
        user_id = 599
        
        # Анализ и визуализация жанров пользователя
        genre_counts = analyze_user_genres(ratings, movies, user_id)
        plot_genre_distribution(genre_counts, user_id)
        
        # Получение и вывод рекомендаций
        get_recommendations(model, ratings, movies, user_id)
        
    except Exception as e:
        logger.error(f"Произошла ошибка в main: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark сессия остановлена")

if __name__ == "__main__":
    main() 