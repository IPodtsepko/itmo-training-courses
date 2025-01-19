import pandas as pd
from sklearn.metrics import mean_squared_error


def mean_squared_error_of(prediction: pd.DataFrame) -> float:
    """ Рассчитывает величину ошибки для предсказания """
    answers = pd.read_csv('./data/answers.csv')
    assert len(prediction) == len(answers), f'Ожидался \'prediction\' размера {len(answers)}'
    return mean_squared_error(answers, prediction)
