import json
import os

import pandas as pd
from celery import Celery
from models_storage import *

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

MONGO_DB_ADDR = os.environ['MONGO_DB_ADDR']
MONGO_DB_PORT = os.environ['MONGO_DB_PORT']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

ml_models_dao = MLModelsDAO(host=MONGO_DB_ADDR,
                            port=MONGO_DB_PORT)

@celery.task(name='add_model')
def add_model (uuid, type_, params):
    try:
        return ml_models_dao.add(uuid, type_, params)
    except AttributeError as a:
        return {'error':a.__str__()}

@celery.task(name='get_all_by_uuid')
def get_all_by_uuid(uuid):
    try:
        return ml_models_dao.get_all_by_uuid(uuid)
    except AttributeError as a:
        return {'error':a.__str__()}

@celery.task(name='delete_user')
def delete_user(uuid):
    ml_models_dao.delete_user(uuid)
    return 'user deleted!'

@celery.task(name='models_available')
def models_available():
    return {'models_types' : list(models_types.keys())}

@celery.task(name='fit_model')
def fit_model(uuid, id, df, target, feat_list, fit_params):
    try:
        ml_models_dao.fit(uuid, id, pd.DataFrame(df), target, feat_list, fit_params)
        return 'Fitted!'
    except AttributeError as a:
        return {'error':a.__str__()}

@celery.task(name='evaluate')
def evaluate(uuid, id, df, metrics):
    try:
        ml_models_dao.evaluate(uuid, id, pd.DataFrame(df), metrics)
        return 'Evaluated!'
    except Exception as ex:
        return {'error':ex.__str__()}

@celery.task(name='predict')
def predict(uuid, id, df, predict_type):
    try:
        preds = getattr(ml_models_dao, predict_type)(uuid, id, pd.DataFrame(df))
        lists = preds.tolist()
        json_str = json.dumps(lists)
        return json_str
    except Exception as ex:
        return {'error': ex.__str__()}

@celery.task(name='update')
def update(uuid, id, params):
    try:
        ml_models_dao.update(uuid, id, params)
        return 'Params updated!'
    except Exception as ex:
        return {'error':ex.__str__()}

@celery.task(name='delete')
def delete(uuid, id):
    try:
        ml_models_dao.delete(uuid, id)
        return 'Model deleted!'
    except Exception as ex:
        return {'error':ex.__str__()}