import sklearn
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import copy
import uuid as uuid64
import pickle

# этот список можно пополнять и дальше в процессе расширения апи :)
models_types = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier' : RandomForestClassifier,    
    'LinearRegression' : LinearRegression,
    'RandomForestRegressor' : RandomForestRegressor,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'KNeighborsClassifier': KNeighborsClassifier,
    'KNeighborsRegressor': KNeighborsRegressor
}

class MLModelsDAO:
    '''
    Класс для хранения моделей и их производных для всех пользователей
    '''
    def __init__(self, host, port = 27017):
        self.__client = MongoClient(host, int(port))
        self.__db = self.__client.mlapi
        self.__models_collection = self.__db.models
        self.__feats_collection = self.__db.feats
    
    def add(self, uuid, type_, params):
        '''
        Добавление модели в общий список
        '''
        if type_ in models_types:
            _id = str(uuid64.uuid4())
            ml_model = {}
            ml_model['uuid'] = uuid
            ml_model['_id'] = _id
            ml_model['type'] = type_
            ml_model['params'] = params
            ml_model['fitted'] = False
            ml_model['instance'] = pickle.dumps(models_types[type_](**params))
            ml_model['metrics'] = {}
            self.__models_collection.insert_one(ml_model)
            return _id
        else:
            raise AttributeError(f'Model type {type_} doesnt exist.')
    
    def get_all_by_uuid(self,uuid):
        '''
        Возвращает список моделей пользователя
        '''
        res = {}
        for model_info in self.__models_collection.find({"uuid": uuid}):
            id_ = model_info['_id']
            del model_info['_id']
            del model_info['instance']
            del model_info['uuid']
            res[id_] = model_info
        return res
    
    def fit(self,uuid, id_, X, target, feat_list, fit_params):
        '''
        Функция обучения и refit модели
        '''
        model_info = self.__models_collection.find_one({'uuid': uuid, '_id': id_})
        if model_info['fitted']:
            #refit model
            params = pickle.loads(model_info['instance']).get_params()
            model = copy.deepcopy(type(pickle.loads(model_info['instance'])))(**params)
            self.__models_collection.update_one({'uuid': uuid, '_id': id_},
                                                {'$set': {'params': params,
                                                          'instance': pickle.dumps(model),
                                                          'fitted': False,
                                                          'metrics': {}}})

        model_info = self.__models_collection.find_one({'uuid': uuid, '_id': id_})
        model = pickle.loads(model_info['instance'])
        model.fit(X[feat_list],X[target],**fit_params)
        if self.__feats_collection.find_one({'uuid': uuid, '_id': id_}) is None:
            self.__feats_collection.insert_one({'uuid': uuid, '_id': id_, 'feats':(feat_list,target)})
        else:
            self.__feats_collection.update_one({'uuid': uuid, '_id': id_},
                                                {'$set': {'feats':(feat_list,target)}})

        self.__models_collection.update_one({'uuid': uuid, '_id': id_},
                                            {'$set': {'instance': pickle.dumps(model),
                                                      'fitted': True}})
        
    def predict(self,uuid,id_,X):
        '''
        Функция предсказания модели
        '''
        model_info = self.__models_collection.find_one({'uuid': uuid, '_id': id_})
        if not model_info['fitted']:
            raise Exception('Model not fitted')

        feat_list,target = self.__feats_collection.find_one({'uuid': uuid, '_id': id_})['feats']
        model = pickle.loads(model_info['instance'])
        pred = model.predict(X[feat_list])

        return pred

    def predict_proba(self,uuid,id_,X):
        '''
        Функция предсказания модели (вероятности)
        '''
        model_info = self.__models_collection.find_one({'uuid': uuid, '_id': id_})
        if not model_info['fitted']:
            raise Exception('Model not fitted')

        feat_list,target = self.__feats_collection.find_one({'uuid': uuid, '_id': id_})['feats']
        model = pickle.loads(model_info['instance'])

        pred = model.predict_proba(X[feat_list])[:,1]

        return pred
        
    def evaluate(self,uuid,id_,X,metrics):
        '''
        Оценка ключевых метрик модели
        '''
        metrics_send = {}
        for metric,predict_func in metrics.items():

            if predict_func == 'predict':
                pred = self.predict(uuid,id_,X)
            elif predict_func == 'predict_proba':
                pred = self.predict_proba(uuid,id_,X)
            else:
                raise AttributeError(f'predict_func only predict or predict_proba')

            _,target = self.__feats_collection.find_one({'uuid': uuid, '_id': id_})['feats']

            metrics_send[metric] = getattr(sklearn.metrics, metric)(X[target], pred)

        self.__models_collection.update_one({'uuid': uuid, '_id': id_},
                                                {'$set': {'metrics': metrics_send}})

    def delete(self,uuid, id_):
        '''
        Удаление моделей из списка
        '''
        self.__models_collection.delete_one({'uuid':uuid,'_id':id_})
        
    def update(self,uuid,id_,params):
        '''
        Измненеие гиперпараметров модели
        '''
        model = copy.deepcopy(type(pickle.loads(self.__models_collection.find_one({'uuid':uuid, '_id':id_})['instance'])))(**params)
        self.__models_collection.update_one({'uuid':uuid, '_id':id_},
                             {'$set': {'params': params,
                                       'instance': pickle.dumps(model),
                                       'fitted': False,
                                       'metrics': {}}})
    
    def delete_user(self,uuid):
        '''
        Удаление пользователя
        '''
        self.__models_collection.delete_many({'uuid': uuid})