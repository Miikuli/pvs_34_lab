#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 3
#define N_TERMS 5000

// Объявления функций
void print_matrix(const char* label, const double matrix[SIZE][SIZE]);
void matrix_add(double result[SIZE][SIZE], 
                const double a[SIZE][SIZE], 
                const double b[SIZE][SIZE]);
void matrix_multiply(double result[SIZE][SIZE],
                     const double a[SIZE][SIZE],
                     const double b[SIZE][SIZE]);
void init_identity_matrix(double matrix[SIZE][SIZE]);
void matrix_scalar_multiply(double result[SIZE][SIZE],
                            const double matrix[SIZE][SIZE],
                            double scalar);
void matrix_sum_operation(void* invec, void* inoutvec, int* len, 
                          MPI_Datatype* datatype);
void compute_partial_sum(const double A[SIZE][SIZE],
                         double partial_sum[SIZE][SIZE],
                         int rank,
                         int num_procs);
void init_matrix(double matrix[SIZE][SIZE], double value);
void copy_matrix(double dest[SIZE][SIZE], const double src[SIZE][SIZE]);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Исходная матрица A
    const double A[SIZE][SIZE] = {
        {0.1, 0.4, 0.2},
        {0.3, 0.0, 0.5},
        {0.6, 0.2, 0.1}
    };
    
    double partial_sum[SIZE][SIZE];    // Частичная сумма для процесса
    double global_sum[SIZE][SIZE];     // Итоговая сумма
    double elapsed_time;               // Время выполнения

    init_matrix(partial_sum, 0.0);
    init_matrix(global_sum, 0.0);

    // Замер времени вычислений
    double start_time = MPI_Wtime();
    compute_partial_sum(A, partial_sum, rank, num_procs);
    
    // Создаем пользовательскую операцию для суммирования матриц
    MPI_Op matrix_sum_op;
    MPI_Op_create(matrix_sum_operation, 1, &matrix_sum_op);
    
    // Собираем результаты на процессе 0
    MPI_Reduce(partial_sum, global_sum, SIZE * SIZE, MPI_DOUBLE,
               matrix_sum_op, 0, MPI_COMM_WORLD);
    
    MPI_Op_free(&matrix_sum_op);
    double end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    // Добавляем единичную матрицу к результату
    if (rank == 0) {
        double identity[SIZE][SIZE];
        init_identity_matrix(identity);
        matrix_add(global_sum, global_sum, identity);
    }

    // Вывод результатов
    if (rank == 0) {
        printf("Matrix size: %dx%d\n", SIZE, SIZE);
        printf("Terms calculated: %d (+ identity matrix)\n", N_TERMS);
        printf("Processes used: %d\n", num_procs);
        printf("Execution time: %.4f seconds\n\n", elapsed_time);
        
        print_matrix("Original matrix A", A);
        print_matrix("Result matrix e^A", global_sum);
    }

    MPI_Finalize();
    return 0;
}

// Вычисление частичной суммы ряда Тейлора
void compute_partial_sum(const double A[SIZE][SIZE],
                         double partial_sum[SIZE][SIZE],
                         int rank,
                         int num_procs) {
    init_matrix(partial_sum, 0.0);
    
    // Предварительный расчет A^P (P = num_procs)
    double A_power[SIZE][SIZE];
    copy_matrix(A_power, A);
    
    double temp[SIZE][SIZE];
    for (int i = 1; i < num_procs; i++) {
        matrix_multiply(temp, A_power, A);
        copy_matrix(A_power, temp);
    }

    // Начальный индекс для текущего процесса
    long start_term = rank + 1;
    
    if (start_term > N_TERMS) 
        return;

    // Инициализация текущего члена ряда
    double current_term[SIZE][SIZE];
    double term_power[SIZE][SIZE];
    
    // Вычисление A^k для первого члена процесса
    init_identity_matrix(term_power);
    for (long p = 1; p <= start_term; p++) {
        matrix_multiply(temp, term_power, A);
        copy_matrix(term_power, temp);
    }

    // Вычисление факториала для первого члена
    double factorial = 1.0;
    for (long p = 2; p <= start_term; p++) {
        factorial *= p;
    }

    // Первое слагаемое для процесса
    matrix_scalar_multiply(current_term, term_power, 1.0 / factorial);
    matrix_add(partial_sum, partial_sum, current_term);

    // Расчет последующих слагаемых
    for (long term = start_term; term <= N_TERMS - num_procs; term += num_procs) {
        long next_term = term + num_procs;
        
        // Умножаем на A^P
        matrix_multiply(temp, current_term, A_power);
        copy_matrix(current_term, temp);
        
        // Вычисление знаменателя: (term+1)*(term+2)*...*(next_term)
        double denominator = 1.0;
        for (long factor = term + 1; factor <= next_term; factor++) {
            denominator *= factor;
        }
        
        // Деление на факториал
        matrix_scalar_multiply(current_term, current_term, 1.0 / denominator);
        
        // Добавление к частичной сумме
        matrix_add(partial_sum, partial_sum, current_term);
    }
}

// Пользовательская операция для суммирования матриц
void matrix_sum_operation(void* invec, void* inoutvec, int* len, 
                          MPI_Datatype* datatype) {
    double (*in)[SIZE] = (double(*)[SIZE])invec;
    double (*out)[SIZE] = (double(*)[SIZE])inoutvec;

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            out[i][j] += in[i][j];
        }
    }
}

// Функции работы с матрицами
void matrix_multiply(double result[SIZE][SIZE],
                     const double a[SIZE][SIZE],
                     const double b[SIZE][SIZE]) {
    double temp[SIZE][SIZE] = {{0}};
    
    for (int i = 0; i < SIZE; i++) {
        for (int k = 0; k < SIZE; k++) {
            for (int j = 0; j < SIZE; j++) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    copy_matrix(result, temp);
}

void matrix_add(double result[SIZE][SIZE], 
                const double a[SIZE][SIZE], 
                const double b[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

void matrix_scalar_multiply(double result[SIZE][SIZE],
                            const double matrix[SIZE][SIZE],
                            double scalar) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = matrix[i][j] * scalar;
        }
    }
}

// Вспомогательные функции
void init_identity_matrix(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void print_matrix(const char* label, const double matrix[SIZE][SIZE]) {
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

void init_matrix(double matrix[SIZE][SIZE], double value) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = value;
        }
    }
}

void copy_matrix(double dest[SIZE][SIZE], const double src[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            dest[i][j] = src[i][j];
        }
    }
}
