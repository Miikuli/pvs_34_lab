import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def parse_output_file(filename):
    """Анализирует файл с результатами и извлекает данные"""
    try:
        with open(filename, 'r') as f:
            content = f.read()

        # Определяем тип и количество процессов
        tech, procs = None, None
        if 'MPI' in filename:
            tech = 'MPI'
            procs = int(re.search(r'MPI_(\d+)_procs', filename).group(1))
        elif 'OMP' in filename:
            tech = 'OpenMP'
            procs = int(re.search(r'OMP_(\d+)_threads', filename).group(1))

        # Извлекаем время выполнения
        time = float(re.search(r'Execution time:\s+([\d.]+)', content).group(1))

        return {'technology': tech, 'processes': procs, 'time': time}

    except Exception as e:
        print(f"Ошибка при анализе файла {filename}: {str(e)}")
        return None


def collect_data(log_dir="logs"):
    """Собирает данные из всех файлов результатов"""
    data = defaultdict(list)
    for filename in glob.glob(os.path.join(log_dir, 'output_*.out')):
        if result := parse_output_file(filename):
            data[(result['technology'], result['processes'])].append(result['time'])
    return data


def calculate_metrics(data):
    """Вычисляет метрики производительности"""
    metrics = {}
    for (tech, procs), times in data.items():
        times = np.array(times)
        base_time = np.mean(data.get((tech, 1), times))

        metrics.setdefault(tech, {
            'processes': [],
            'time_mean': [],
            'time_std': [],
            'speedup': [],
            'efficiency': []
        })

        metrics[tech]['processes'].append(procs)
        metrics[tech]['time_mean'].append(np.mean(times))
        metrics[tech]['time_std'].append(np.std(times))
        metrics[tech]['speedup'].append(base_time / np.mean(times))
        metrics[tech]['efficiency'].append((base_time / np.mean(times)) / procs)

    # Сортируем по количеству процессов
    for tech in metrics:
        order = np.argsort(metrics[tech]['processes'])
        for key in metrics[tech]:
            metrics[tech][key] = np.array(metrics[tech][key])[order]

    return metrics


def plot_results(metrics):
    """Строит и сохраняет графики"""
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(15, 10))

    colors = {'MPI': 'blue', 'OpenMP': 'red'}

    # График 1: Время выполнения
    plt.subplot(2, 2, 1)
    for tech, data in metrics.items():
        plt.errorbar(data['processes'], data['time_mean'], yerr=data['time_std'],
                     fmt='-o', label=tech, color=colors[tech], capsize=5)
    plt.xlabel('Число процессов/потоков')
    plt.ylabel('Время (с)')
    plt.title('Сравнение времени выполнения')
    plt.legend()
    plt.grid(True)

    # График 2: Ускорение
    plt.subplot(2, 2, 2)
    for tech, data in metrics.items():
        plt.plot(data['processes'], data['speedup'], '-o',
                 label=tech, color=colors[tech])
    plt.plot([1, max(data['processes'])], [1, max(data['processes'])],
             'k--', label='Линейное ускорение')
    plt.xlabel('Число процессов/потоков')
    plt.ylabel('Ускорение')
    plt.title('Сравнение ускорения')
    plt.legend()
    plt.grid(True)

    # График 3: Эффективность
    plt.subplot(2, 2, 3)
    for tech, data in metrics.items():
        plt.plot(data['processes'], data['efficiency'], '-o',
                 label=tech, color=colors[tech])
    plt.axhline(1, color='k', linestyle='--')
    plt.xlabel('Число процессов/потоков')
    plt.ylabel('Эффективность')
    plt.title('Сравнение эффективности')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/comparison.png')
    plt.show()


def save_to_excel(metrics):
    """Сохраняет данные в Excel"""
    df_data = []
    for tech, data in metrics.items():
        for i in range(len(data['processes'])):
            df_data.append({
                'Technology': tech,
                'Processes': data['processes'][i],
                'Time (s)': data['time_mean'][i],
                'Time Std': data['time_std'][i],
                'Speedup': data['speedup'][i],
                'Efficiency': data['efficiency'][i]
            })

    pd.DataFrame(df_data).to_excel('results/metrics.xlsx', index=False)


def main():
    print("Начинаем анализ производительности...")

    # Шаг 1: Сбор данных
    print("Сбор данных из логов...")
    data = collect_data("C:/Users/Кукуруза/PycharmProjects/pythonProject7/logs")

    if not data:
        print("Ошибка: не найдены файлы результатов в ../logs/")
        return

    # Шаг 2: Расчет метрик
    print("Расчет метрик...")
    metrics = calculate_metrics(data)

    # Шаг 3: Визуализация
    print("Построение графиков...")
    plot_results(metrics)

    # Шаг 4: Сохранение
    print("Сохранение результатов...")
    save_to_excel(metrics)

    print("Анализ завершен! Результаты в папке results/")


if __name__ == "__main__":
    main()
