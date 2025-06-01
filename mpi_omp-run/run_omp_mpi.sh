#!/bin/bash

# Конфигурация скрипта
LOG_DIR="logs"
REPEATS=10
PROCESS_CONFIG=(1 2 4 8 16)  # Конфигурации процессов/потоков
MPI_EXEC="exp_mpi"
OMP_EXEC="exp_omp"
MPI_SRC="${MPI_EXEC}.c"
OMP_SRC="${OMP_EXEC}.c"

# Настройки окружения OpenMP
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=dynamic,64

# Загрузка необходимых модулей
module load mpi/openmpi-x86_64

# Функция для очистки и создания директорий
setup_environment() {
    echo "Подготовка окружения..."
    rm -rf "$LOG_DIR"
    mkdir -p "$LOG_DIR"
}

# Функция компиляции программ
compile_programs() {
    echo "Компиляция MPI программы ($MPI_SRC)..."
    mpicc -o "$MPI_EXEC" "$MPI_SRC" -lm -O3 || exit 1
    
    echo "Компиляция OpenMP программы ($OMP_SRC)..."
    gcc -fopenmp -o "$OMP_EXEC" "$OMP_SRC" -lm -O3 || exit 1
}

# Функция определения параметров задания
get_job_params() {
    local processes=$1
    local type=$2
    local walltime=""
    local mem="2GB"  # Фиксированный объем памяти

    # Определение времени выполнения в зависимости от типа и количества процессов
    if [[ "$type" == "MPI" ]]; then
        if [ "$processes" -le 2 ]; then
            walltime="00:05"
        elif [ "$processes" -le 8 ]; then
            walltime="00:10"
        else
            walltime="00:20"
        fi
    else  # OpenMP
        if [ "$processes" -le 2 ]; then
            walltime="01:00"
        elif [ "$processes" -le 8 ]; then
            walltime="02:10"
        else
            walltime="03:20"
        fi
    fi
    
    echo "$walltime $mem"
}

# Функция запуска MPI заданий
run_mpi_jobs() {
    echo "Запуск MPI задач..."
    for procs in "${PROCESS_CONFIG[@]}"; do
        read -r walltime mem <<< $(get_job_params "$procs" "MPI")
        
        for ((run=1; run<=REPEATS; run++)); do
            job_name="MPI_${procs}_procs_run${run}"
            
            bsub <<JOB
#!/bin/bash
#BSUB -J $job_name
#BSUB -W $walltime
#BSUB -n $procs
#BSUB -R "span[ptile=$procs]"
#BSUB -o ${LOG_DIR}/output_${job_name}_%J.out
#BSUB -e ${LOG_DIR}/error_${job_name}_%J.err
#BSUB -M $mem

module load mpi/openmpi-x86_64
mpirun --bind-to core --map-by core ./$MPI_EXEC
JOB

            sleep 0.2  # Пауза между отправкой заданий
        done
    done
}

# Функция запуска OpenMP заданий
run_omp_jobs() {
    echo "Запуск OpenMP задач..."
    for threads in "${PROCESS_CONFIG[@]}"; do
        read -r walltime mem <<< $(get_job_params "$threads" "OMP")
        
        for ((run=1; run<=REPEATS; run++)); do
            job_name="OMP_${threads}_threads_run${run}"
            
            bsub <<JOB
#!/bin/bash
#BSUB -J $job_name
#BSUB -W $walltime
#BSUB -n 1  # Всегда 1 процесс для OpenMP
#BSUB -o ${LOG_DIR}/output_${job_name}_%J.out
#BSUB -e ${LOG_DIR}/error_${job_name}_%J.err
#BSUB -M $mem

export OMP_NUM_THREADS=$threads
./$OMP_EXEC
JOB

            sleep 0.2  # Пауза между отправкой заданий
        done
    done
}

# Основной поток выполнения
main() {
    setup_environment
    compile_programs
    run_mpi_jobs
    run_omp_jobs
    echo "Все задачи успешно отправлены в очередь LSF"
}

# Запуск основной функции
main
