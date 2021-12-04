from celery import Celery
from flask import Flask, session, escape
from flask_restx import Api, Resource
from datsets_storage import *
import os
import uuid as uuid64

app = Flask(__name__)
api = Api(app)

app.secret_key = os.urandom(24)

CELERY_BROKER = os.environ['CELERY_BROKER']
CELERY_BACKEND = os.environ['CELERY_BACKEND']

MONGO_DB_ADDR = os.environ['MONGO_DB_ADDR']
MONGO_DB_PORT = os.environ['MONGO_DB_PORT']

celery = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_BACKEND)

dset_storage = DatasetStorage(host=MONGO_DB_ADDR,
                            port=MONGO_DB_PORT)

@app.route('/api/login', methods = ['GET'])
def login():
    """
    Для создание индивидуальной сессии. Сессией считается экземпляр класса MLApi на стороне клиента
    """
    if len(session) == 0:
        session['uuid'] = uuid64.uuid4().__str__()
    return ''

@app.route('/api/logout', methods=['GET'])
def logout():
    """
    Удаление всех файлов при окончании сессии
    """
    if len(session) != 0:
        uuid = session.pop('uuid', None)
        dset_storage.delete_user(uuid)

        celery.send_task('delete_user', args=[escape(session['uuid'])])

    return ''

@app.route('/api/task_result', methods=['GET'])
def results():
    """
    Возвращает результат выполнения task по task_id
    """
    task = celery.AsyncResult(api.payload['task_id'])

    if task.state == "PENDING":
        return task.state
    else:
        res = task.get()
        if isinstance(res,dict):
            if 'error' in res.keys():
                return res['error'], 500
            else:
                return res, 200
        else:
            return res, 200

@app.route('/api/datasets/<dataset_type>', methods=['POST'])
def upload_files(dataset_type):
    """
    Функция для загрузки датасета. Если в хранилище уже существует датасет с типом <dataset_type>, то произойдет перезапись.
    """
    if len(session) == 0:
        return 'you are not authorized', 401
    if dataset_type not in ['train','eval','test']:
        return 'Only train, eval, test value in type_.', 400
    try:
        df = api.payload
        uuid = escape(session['uuid'])
        dset_storage.write(uuid,dataset_type,df)
    except Exception as ex:
        return ex.__str__(), 500
    return ''

@app.route('/api/ml_models/available', methods=['GET'])
def models_available():
    """
    Выводит список доступных на сервере моделей
    """
    if len(session) == 0:
        return 'you are not authorized', 401
    task_id = celery.send_task('models_available')
    return {'task_id': str(task_id)}, 200
    # except Exception as ex:
    #     return ex.__str__(), 500

@api.route('/api/ml_models')
class MLModels(Resource):
    '''
    Класс для создания, получения ml моделей
    '''

    def get(self):
        if len(session) == 0:
            return 'you are not authorized', 401

        task_id = celery.send_task('get_all_by_uuid', args=[escape(session['uuid'])])
        return {'task_id': str(task_id)}, 200
        # except Exception as ex:
        #     return ex.__str__(), 500

    def post(self):
        if len(session) == 0:
            return 'you are not authorized', 401

        task_id = celery.send_task('add_model',
                                        args=[escape(session['uuid']),
                                              api.payload['model_type'],
                                              api.payload['params']])

        return {'task_id': str(task_id)}, 200
        # except Exception as ex:
        #     return ex.__str__(), 500

@app.route('/api/fit/<id>', methods=['POST'])
def fit_model(id):
    '''
    функция обучения модели
    '''
    if len(session) == 0:
        return 'you are not authorized', 401

    try:
        target = api.payload['target']
        feat_list = api.payload['feat_list']
        fit_params = api.payload['fit_params']
        uuid = escape(session['uuid'])

        df = dset_storage.get(uuid, 'train')
    except Exception as ex:
        return ex.__str__(), 500

    task_id = celery.send_task('fit_model', args=[uuid, id, df, target, feat_list, fit_params])
    return {'task_id': str(task_id)}, 200

@app.route('/api/evaluate/<id>', methods=['POST'])
def evaluate(id):
    '''
    функция для оценки модели по ключевым метрикам
    '''
    if len(session) == 0:
        return 'you are not authorized', 401

    try:
        metrics = api.payload['metrics']
        uuid = escape(session['uuid'])
        df = dset_storage.get(uuid, 'eval')
    except Exception as ex:
        return ex.__str__(), 500

    task_id = celery.send_task('evaluate', args=[uuid, id, df, metrics])
    return {'task_id': str(task_id)}, 200

@app.route('/api/predict/<id>', methods=['GET'])
def predict(id):
    '''
    Функция возвращает предсказания модели
    '''
    if len(session) == 0:
        return 'you are not authorized', 401

    try:
        sample_type = api.payload['sample_type']
        predict_type = api.payload['predict_type']
        uuid = escape(session['uuid'])

        df = dset_storage.get(uuid, sample_type)
    except Exception as ex:
        return ex.__str__(), 500

    task_id = celery.send_task('predict', args=[uuid, id, df, predict_type])
    return {'task_id': str(task_id)}, 200


@api.route('/api/ml_models/<id>')
class MLModelsID(Resource):
    '''
    Класс для редактирования гиперпарметров модели и для удаления модели
    '''

    def put(self, id):
        if len(session) == 0:
            return 'you are not authorized', 401

        try:
            uuid = escape(session['uuid'])
        except Exception as ex:
            return ex.__str__(), 500

        task_id = celery.send_task('update', args=[uuid, id, api.payload['params']])
        return {'task_id': str(task_id)}, 200

    def delete(self, id):
        if len(session) == 0:
            return 'you are not authorized', 401

        try:
            uuid = escape(session['uuid'])
        except Exception as ex:
            return ex.__str__(), 500

        task_id = celery.send_task('delete', args=[uuid, id])
        return {'task_id': str(task_id)}, 200


if __name__ == '__main__':
    app.run(port=os.environ['PORT'],
            host=os.environ['HOST'],
            debug=os.environ['DEBUG'])