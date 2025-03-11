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
        
        # Вывод названий столбцов
        print("\nНазвания столбцов:")
        print(df.columns.tolist())
        
        return df
    
    except Exception as e:
        logging.error(f"Ошибка при чтении файла: {e}")
        return None

# Основная функция
def main():
    df = read_criteo_dataset(train_path)

if __name__ == "__main__":
    main()

