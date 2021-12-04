import sys

import tqdm
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
import time

class MlApi ():
    
    def __init__(self, host= 'http://127.0.0.1:5000'):
        self.__session = requests.Session()
        self.__host = host
        self.__models = {}
        self.__login()
    
    def __login(self):
        """
        Для создание индивидуальной сессии. Сессией считается экземпляр класса MLApi
        """
        r = self.__session.get(f'{self.__host}/api/login')
        if r.status_code != 200:
            raise Exception(r.text)

    def __task_execution(self, r):
        time.sleep(1)
        if r.status_code in [200,204]:
            res = self.__session.get(f'{self.__host}/api/task_result', json={'task_id': r.json()['task_id']})
            if res.text == 'PENDING':
                print(f'Your task in progress. Task id = {r.json()["task_id"]}. ' \
                       f'Check result later. For checking use result() function.')
                sys.exit(0)
            elif res.status_code not in [200,204]:
                raise Exception(res.text)
            else:
                return res
        else:
            raise Exception(r.text)
        
    
    def load_dataset(self,type_, data: pd.core.frame.DataFrame):
        """
        Функция для загрузки датасета в api
        """
        print('data uploading...')
        r = self.__session.post(f'{self.__host}/api/datasets/{type_}', json=data.to_dict())
        if r.status_code == 200:
            print('data upload')
        else:
            raise Exception(r.text)

    def models_available(self):
        """
        Выводит список доступных моделей на сервере
        """
        r = self.__session.get(f'{self.__host}/api/ml_models/available')
        r = self.__task_execution(r)
        return r.json()['models_types']
            
    def add_model(self, model_name, model_type, params={}):
        """
        Добавление новой модели
        """
        
        if model_name in self.__models.keys():
            raise Exception(f'Модель {model_name} уже существует')
        r = self.__session.post(f'{self.__host}/api/ml_models', json = {'model_type':model_type, 'params':params})
        r = self.__task_execution(r)
        self.__models[model_name] = r.text
        print(f"{model_name} added")
    
    def gel_model_list(self):
        """
        Выводит весь список доступных моделей
        """
        r = self.__session.get(f'{self.__host}/api/ml_models')
        revers_models = {v:k for k,v in self.__models.items()}
        if r.status_code == 200:
            r = self.__task_execution(r)
            model_list = defaultdict(list)
            models = r.json()
            for id_, model in models.items():
                model_list['Model name'].append(revers_models[id_])
                for k,v in model.items():
                    if k!='metrics':
                        model_list[k].append(v)
            if len(models) !=0 :
                return pd.DataFrame(model_list)
            else:
                print('Список моделей пуст')
        else:
            print(r.text)
            
    def result(self,task_id):
        r = self.__session.get(f'{self.__host}/api/task_result', json={'task_id':task_id})
        if r.status_code == 200:
            try:
                return r.json()
            finally:
                return r.text
        else:
            raise Exception(r.text)
            
    def fit_model(self, model_name, feat_list, target_name, fit_params={}):
        """
        Функция для обучения модели 
        """
        id_ = self.__models[model_name]
        r = self.__session.post(f'{self.__host}/api/fit/{id_}', json = {'feat_list':feat_list, 
                                                                                    'target':target_name, 
                                                                                   'fit_params':fit_params})
        r = self.__task_execution(r)
        print(r.text)
            
    def evaluate_model(self, model_name, metrics):
        """
        Функция для оценки модели по ключевым метрикам 
        """
        id_ = self.__models[model_name]
        print(f'Evaluating {model_name}...')
        r = self.__session.post(f'{self.__host}/api/evaluate/{id_}', json = {'metrics':metrics})
        r = self.__task_execution(r)
        print(r.text)
    
    def predict_model(self, model_name, predict_type = 'predict', sample_type='test'):
        """
        Возвращает предсказания модели для sample_type
        """
        id_ = self.__models[model_name]
        r = self.__session.get(f'{self.__host}/api/predict/{id_}', json = {'sample_type':sample_type,
                                                                           'predict_type':predict_type})
        r = self.__task_execution(r)
        return np.array(r.json())
            
    def get_evaluation_results(self):
        """
        Возвраещет результаты функции оценки модели по ключевым метрикам
        """
        r = self.__session.get(f'{self.__host}/api/ml_models')
        r = self.__task_execution(r)
        revers_models = {v:k for k,v in self.__models.items()}
        if r.status_code == 200:
            models_res = {}
            models = r.json()
            for id_, model in models.items():
                models_res[revers_models[id_]] = model['metrics']
                
            return models_res
        else:
            print(r.text)
    
    def delete_model(self, model_name):
        """
        Удаляет модель из списков
        """
        id_ = self.__models[model_name]
        r = self.__session.delete(f'{self.__host}/api/ml_models/{id_}')
        _ = self.__task_execution(r)
        del self.__models[model_name]
        print(f'Model {model_name} deleted')
    
    def change_hyperparams(self, model_name, params):
        """
        Функция заменяет текущие гипперпараметры модели на новые. !! После замены придется заново обучить модель !!
        """
        id_ = self.__models[model_name]
        r = self.__session.put(f'{self.__host}/api/ml_models/{id_}', json = {'params':params})
        r = self.__task_execution(r)
        print(r.text)
            
    def __del__(self):
        """
        Удаление всех данных сессии
        """
        self.__session.get(f'{self.__host}/api/logout')