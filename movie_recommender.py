from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, explode, split, count, desc, regexp_extract
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import shutil
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session():
    #Создание Spark сессии
    try:
        return SparkSession.builder \
            .appName("MovieRecommender") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.local.dir", "./spark_temp") \
            .config("spark.memory.fraction", "0.6") \
            .config("spark.memory.storageFraction", "0.5") \
            .config("spark.shuffle.file.buffer", "512k") \
            .config("spark.shuffle.spill.compress", "true") \
            .config("spark.shuffle.compress", "true") \
            .config("spark.broadcast.compress", "true") \
            .config("spark.sql.shuffle.partitions", "50") \
            .config("spark.executor.cores", "2") \
            .config("spark.task.cpus", "1") \
            .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
            .config("spark.eventLog.gcMetrics.youngGenerationGarbageCollectors", "G1 Young Generation") \
            .config("spark.eventLog.gcMetrics.oldGenerationGarbageCollectors", "G1 Old Generation") \
            .config("spark.network.timeout", "600s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .getOrCreate()
    except Exception as e:
        logger.error(f"Ошибка при создании Spark сессии: {str(e)}")
        raise

def load_data(spark, data_path):
    #Загрузка данных из DAT файлов
    try:
        # Загрузка фильмов
        movies = spark.read.text(os.path.join(data_path, "movies.dat")) \
            .select(
                regexp_extract("value", r"^(\d+)::", 1).cast("integer").alias("movieID"),
                regexp_extract("value", r"^\d+::(.*?)::", 1).alias("title"),
                regexp_extract("value", r"::([^:]+)$", 1).alias("genres")
            ) \
            .repartition(10) \
            .persist()
        
        # Загрузка оценок с ограничением для тестирования
        ratings = spark.read.text(os.path.join(data_path, "ratings.dat")) \
            .select(
                regexp_extract("value", r"^(\d+)::", 1).cast("integer").alias("userID"),
                regexp_extract("value", r"^\d+::(\d+)::", 1).cast("integer").alias("movieID"),
                regexp_extract("value", r"::(\d+\.\d+)::", 1).cast("float").alias("rating"),
                regexp_extract("value", r"::(\d+)$", 1).cast("long").alias("timestamp")
            ) \
            .filter(col("rating") > 0) #\
            #.sample(fraction=0.95, seed=42)  # Уменьшаем до 20% данных
        ratings = ratings.repartition(20).persist()
            
        # Удаление пользователей с менее чем 5 оценками
        user_counts = ratings.groupBy("userID").count()
        valid_users = user_counts.filter(col("count") >= 5).select("userID")
        ratings = ratings.join(valid_users, on="userID").persist()
        
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
            rank=20,  # Увеличиваем rank для большей точности
            userCol="userID",
            itemCol="movieID",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True  # Добавляем nonnegative для большей точности
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
        # Создаем папку для графиков, если её нет
        os.makedirs("graphs", exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=ratings_pd, x="rating", bins=20)
        plt.title("Распределение оценок фильмов")
        plt.xlabel("Оценка")
        plt.ylabel("Количество")
        plt.savefig("graphs/ratings_distribution.png")
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
        # Создаем папку для графиков, если её нет
        os.makedirs("graphs", exist_ok=True)
        
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
        plt.savefig(f"graphs/user_{user_id}_genres.png")
        plt.close()
        
        logger.info(f"График распределения жанров сохранен для пользователя {user_id}")
    except Exception as e:
        logger.error(f"Ошибка при создании графика жанров: {str(e)}")

def get_recommendations(model, ratings_df, movies_df, user_id, n_candidates=200, n_recommendations=20):
    #Получение рекомендаций для пользователя
    try:
        # Получаем больше кандидатов для разнообразия
        user_subset = ratings_df.filter(col("userID") == user_id).select(["userID", "movieID"])
        user_recs = model.recommendForUserSubset(user_subset, n_candidates)
        
        # Получаем список уже просмотренных фильмов пользователем
        watched_movies = ratings_df.filter(col("userID") == user_id) \
            .select("movieID") \
            .collect()
        watched_movie_ids = {row.movieID for row in watched_movies}
        
        # Собираем все рекомендации
        all_recommendations = []
        for row in user_recs.collect():
            for movie_id, rating in row.recommendations:
                # Пропускаем уже просмотренные фильмы
                if movie_id not in watched_movie_ids:
                    movie_info = movies_df.filter(col("movieID") == movie_id).first()
                    if movie_info:
                        # Ограничиваем рейтинг до 5.0
                        normalized_rating = min(float(rating), 5.0)
                        all_recommendations.append({
                            'movieID': movie_id,
                            'title': movie_info.title,
                            'genres': movie_info.genres,
                            'rating': normalized_rating
                        })
        
        # Сортируем по рейтингу и берем случайные n_recommendations из топ-50%
        if all_recommendations:
            # Сортируем по рейтингу
            all_recommendations.sort(key=lambda x: x['rating'], reverse=True)
            
            # Берем топ-50% фильмов
            top_half = all_recommendations[:len(all_recommendations)//2]
            
            # Выбираем случайные n_recommendations из топ-50%
            random.seed(42)  # Для воспроизводимости
            selected_recommendations = random.sample(top_half, min(n_recommendations, len(top_half)))
            
            print(f"\nРекомендации для пользователя {user_id}:")
            print("\n{:<60} {:<30} {:<10}".format("Название фильма", "Жанры", "Рейтинг"))
            print("-" * 100)
            
            # Собираем все жанры из рекомендаций для анализа
            recommended_genres = []
            
            for rec in selected_recommendations:
                title = rec['title']
                genres = rec['genres']
                rating = rec['rating']  # Уже нормализованный рейтинг
                recommended_genres.extend(genres.split("|"))
                
                # Обрезаем длинные строки
                if len(title) > 57:
                    title = title[:54] + "..."
                if len(genres) > 27:
                    genres = genres[:24] + "..."
                    
                print("{:<60} {:<30} {:<10.2f}".format(title, genres, rating))
            
            # Анализируем и визуализируем жанры рекомендаций
            if recommended_genres:
                # Создаем папку для графиков, если её нет
                os.makedirs("graphs", exist_ok=True)
                
                genre_counts = pd.Series(recommended_genres).value_counts()
                plt.figure(figsize=(12, 6))
                genre_counts.plot(kind='bar')
                plt.title(f"Жанры в рекомендациях для пользователя {user_id}")
                plt.xlabel("Жанр")
                plt.ylabel("Количество фильмов")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"graphs/user_{user_id}_recommended_genres.png")
                plt.close()
        else:
            print(f"Не удалось найти рекомендации для пользователя {user_id}")
                    
    except Exception as e:
        logger.error(f"Ошибка при получении рекомендаций: {str(e)}")
        raise

def main():
    try:
        # Создание директории для временных файлов
        temp_dir = "./spark_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Создание Spark сессии
        spark = create_spark_session()
        
        # Установка уровня логирования
        spark.sparkContext.setLogLevel("WARN")
        
        # Загрузка данных
        ratings, movies = load_data(spark, "ml-10M100K")
        
        try:
            # Разделение данных на обучающую и тестовую выборки
            train, test = ratings.randomSplit([0.8, 0.2], seed=13)
            
            # Обучение модели
            model = train_model(train)
            
            # Оценка модели
            rmse = evaluate_model(model, test)
            
            # Визуализация распределения оценок
            ratings_pd = ratings.sample(fraction=0.5, seed=42).toPandas()
            plot_ratings_distribution(ratings_pd)
            
            # Пример получения рекомендаций для пользователя
            user_id = 558
            
            # Анализ и визуализация жанров пользователя
            genre_counts = analyze_user_genres(ratings, movies, user_id)
            plot_genre_distribution(genre_counts, user_id)
            
            # Получение и вывод рекомендаций
            get_recommendations(model, ratings, movies, user_id, n_candidates=100, n_recommendations=20)
            
        finally:
            # Очистка кэша после использования
            ratings.unpersist()
            movies.unpersist()
            
    except Exception as e:
        logger.error(f"Произошла ошибка в main: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark сессия остановлена")
            # Очистка временных файлов
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.error(f"Ошибка при очистке временных файлов: {str(e)}")

if __name__ == "__main__":
    main() 