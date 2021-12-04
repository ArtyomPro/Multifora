from pymongo import MongoClient
import pandas as pd


class DatasetStorage:
    '''
    Класс для хранения датасетов всех пользователей
    '''
    def __init__(self, host, port = 27017):
        self.__client = MongoClient(host, int(port))
        self.__db = self.__client.mlapi
        self.__datasets_collection = self.__db.datasets

    def write(self, uuid, type_, df):
        if self.__datasets_collection.find_one({"uuid": uuid, "type":type_}) is None:
            dset = {}
            dset['uuid'] = uuid
            dset['type'] = type_
            dset['df'] = df
            self.__datasets_collection.insert_one(dset)
        else:
            self.__datasets_collection.update_one({'uuid': uuid, "type":type_},
                                                {'$set': {'df': df}})

    def get(self, uuid, type_):
        df = self.__datasets_collection.find_one({"uuid": uuid, "type":type_})['df']
        return df

    def delete_user(self, uuid):
        self.__datasets_collection.delete_many({'uuid': uuid})