import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Путь к файлу
train_path = '/home/users/datasets/criteo/train1000.txt'

# Параметры чтения файла
def read_criteo_dataset(path):
    try:
        # Чтение TSV файла без заголовка
        df = pd.read_csv(path, sep='\t', header=None)
        
        # Вывод базовой информации о датасете
        print("Базовая информация о датасете:")
        print(f"Количество строк: {len(df)}")
        print(f"Количество столбцов: {len(df.columns)}")
        
        # Вывод первых строк
        print("\nПервые 5 строк:")
        print(df.head())
        
        # Вывод типов данных
        print("\nТипы данных:")
        print(df.dtypes)
        
        # Вывод краткой статистики
        print("\nКраткая статистика:")
        print(df.describe())
        
        return df
    
    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        return None

# Основная функция
def main():
    df = read_criteo_dataset(train_path)

if __name__ == "__main__":
    main()
