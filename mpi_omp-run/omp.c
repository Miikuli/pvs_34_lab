#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 3
#define N_TERMS 5000

// Объявления функций
void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]);
void MatrixAdd(double result[SIZE][SIZE], 
              const double matrix1[SIZE][SIZE],
              const double matrix2[SIZE][SIZE]);
void MatrixMultiply(double result[SIZE][SIZE],
                   const double matrix1[SIZE][SIZE],
                   const double matrix2[SIZE][SIZE]);
void MatrixIdentity(double matrix[SIZE][SIZE]);
void MatrixScalarMultiply(double result[SIZE][SIZE],
                         const double matrix[SIZE][SIZE],
                         const double scalar);
void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]);
void MatrixPower(double result[SIZE][SIZE], const double matrix[SIZE][SIZE], int power);

int main() {
    double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2}, 
        {0.3, 0.0, 0.5}, 
        {0.6, 0.2, 0.1}
    };
    double taylor_sum[SIZE][SIZE] = {0};  // Сумма ряда
    double identity[SIZE][SIZE];          // Единичная матрица
    double start_time, end_time;          // Для замера времени

    start_time = omp_get_wtime();

    // ПАРАЛЛЕЛЬНАЯ ОБЛАСТЬ
    #pragma omp parallel
    {
        double local_sum[SIZE][SIZE] = {0};  // Локальная сумма для потока
        int tid = omp_get_thread_num();      // ID потока
        int num_threads = omp_get_num_threads(); // Общее число потоков
        
        // Вычисляем A^P (P = num_threads) - общий шаг для потока
        double A_step[SIZE][SIZE];
        MatrixPower(A_step, A, num_threads);
        
        // Начальный индекс k для этого потока
        int start_k = tid + 1;
        
        if (start_k <= N_TERMS) {
            // ВЫЧИСЛЯЕМ ПЕРВОЕ СЛАГАЕМОЕ ДЛЯ ПОТОКА -------
            
            // 1. Вычисляем A^start_k
            double A_power[SIZE][SIZE];
            MatrixPower(A_power, A, start_k);
            
            // 2. Вычисляем факториал start_k!
            double factorial = 1.0;
            for (int i = 1; i <= start_k; i++) {
                factorial *= i;
            }
            
            // 3. Формируем первое слагаемое: A^start_k / start_k!
            double term[SIZE][SIZE];
            MatrixScalarMultiply(term, A_power, 1.0 / factorial);
            MatrixAdd(local_sum, local_sum, term);
            
            // 4. Сохраняем текущее слагаемое для рекуррентных вычислений
            double current_term[SIZE][SIZE];
            MatrixCopy(current_term, term);
            
            // РЕКУРРЕНТНОЕ ВЫЧИСЛЕНИЕ ПОСЛЕДУЮЩИХ СЛАГАЕМЫХ ----
            for (int k = start_k + num_threads; k <= N_TERMS; k += num_threads) {
                // 1. Умножаем текущее слагаемое на A^step
                double temp[SIZE][SIZE];
                MatrixMultiply(temp, current_term, A_step);
                MatrixCopy(current_term, temp);
                
                // 2. Вычисляем дополнительный множитель для факториала:
                //    (k - num_threads + 1) * (k - num_threads + 2) * ... * k
                double factor = 1.0;
                for (int j = k - num_threads + 1; j <= k; j++) {
                    factor *= j;
                }
                
                // 3. Делим на дополнительный множитель
                MatrixScalarMultiply(current_term, current_term, 1.0 / factor);
                
                // 4. Добавляем к локальной сумме
                MatrixAdd(local_sum, local_sum, current_term);
            }
        }
        
        // ОБЪЕДИНЯЕМ РЕЗУЛЬТАТЫ ПОТОКОВ (критическая секция)
        #pragma omp critical
        {
            MatrixAdd(taylor_sum, taylor_sum, local_sum);
        }
    }
    
    // ДОБАВЛЯЕМ ЕДИНИЧНУЮ МАТРИЦУ (k=0)
    MatrixIdentity(identity);
    MatrixAdd(taylor_sum, taylor_sum, identity);

    end_time = omp_get_wtime();

    // ВЫВОД РЕЗУЛЬТАТОВ (только главный поток)
    printf("Matrix size: %dx%d\n", SIZE, SIZE);
    printf("Number of terms: %d (+ Identity)\n", N_TERMS);
    printf("Number of threads: %d\n", omp_get_max_threads());
    printf("Execution time: %.4f seconds\n\n", end_time - start_time);
    
    MatrixPrint("Original matrix A", A);
    MatrixPrint("Result matrix e^A", taylor_sum);

    return 0;
}

// Функция возведения матрицы в степень
void MatrixPower(double result[SIZE][SIZE], const double matrix[SIZE][SIZE], int power) {
    double temp[SIZE][SIZE];
    MatrixIdentity(result);  // Начинаем с единичной матрицы
    
    // Последовательное умножение для возведения в степень
    for (int p = 0; p < power; p++) {
        MatrixMultiply(temp, result, matrix);
        MatrixCopy(result, temp);
    }
}

void MatrixPrint(const char* label, const double matrix[SIZE][SIZE]) {
    printf("%s:\n", label);
    for (int i = 0; i < SIZE; i++) {
        printf("  [");
        for (int j = 0; j < SIZE; j++) {
            printf("%8.4f", matrix[i][j]);
            if (j < SIZE - 1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

void MatrixAdd(double result[SIZE][SIZE], 
              const double matrix1[SIZE][SIZE],
              const double matrix2[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void MatrixMultiply(double result[SIZE][SIZE],
                   const double matrix1[SIZE][SIZE],
                   const double matrix2[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void MatrixIdentity(double matrix[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void MatrixScalarMultiply(double result[SIZE][SIZE],
                         const double matrix[SIZE][SIZE],
                         const double scalar) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
}

void MatrixCopy(double dest[SIZE][SIZE], const double src[SIZE][SIZE]) {
    #pragma omp simd collapse(2)
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            dest[i][j] = src[i][j];
        }
    }
}
