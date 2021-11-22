# Deer-detection
Repository on ROSNEFT hackaton deer population task
Team: Repeat

# Шаблон решения

## Структура проекта
Проект должен иметь следующую структуру. Обязательно наличие *Dockerfile*, *requirements.txt*, *run.py* и 
весов модели в папке *work_dir/pretrained*, если используются.
Тестовые картинки лежат в папке *work_dir/input*.
Базовый докер-образ может быть любым.
```
.
├── Dockerfile
├── notebooks
├── README.md
├── requirements.txt
├── run.py
└── work_dir
    ├── modules
    ├── configs
    ├── input
    │   ├── img0.jpg
    │   ├── img1.jpg
    │   ...
    │   └── imgn.jpg
    ├── output
    │   └── submission.json
    └── pretrained
        └── model.weights
```
## Запуск модели
Запуск модели производится с вызовом модуля *run.py* с параметром *--mode predict*.

## Сабмишен файл
Результатом тестирования является файл *work_dir/output/submission.csv*.
