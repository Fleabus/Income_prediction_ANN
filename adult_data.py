
import tensorflow as tf
import numpy as np
import pandas as pd

class Adult_Data:
    workclass_categories = None
    education_categories = None
    marital_status_categories = None
    occupation_categories = None
    relationship_categories = None
    race_categories = None
    sex_categories = None
    native_country_categories = None
    income_categories = None

    def __init__ (self):
        self.train = pd.read_csv('train.csv')
        self.test = pd.read_csv('test.csv')
        self.train = self.data_altering(self.train)
        self.test = self.data_altering(self.test)

    def get_len(self, type):
        if(type == "train"):
            return len(self.train)
        elif(type == "test"):
            return len(self.test)
        else:
            return None

    def data_altering(self, data):
        data['age'] = self.format_age(data['age'])
        data['workclass'] = self.format_workclass(data['workclass'])
        data['fnlwgt'] = self.format_fnlwgt(data['fnlwgt'])
        data['education'] = self.format_education(data['education'])
        data['education-num'] = self.format_education_num(data['education-num'])
        data['marital-status'] = self.format_marital_status(data['marital-status'])
        data['occupation'] = self.format_occupation(data['occupation'])
        data['relationship'] = self.format_relationship(data['relationship'])
        data['race'] = self.format_race(data['race'])
        data['sex'] = self.format_sex(data['sex'])
        data['capital-gain'] = self.format_capital_gain(data['capital-gain'])
        data['capital-loss'] = self.format_capital_loss(data['capital-loss'])
        data['hours-per-week'] = self.format_hours_per_week(data['hours-per-week'])
        data['native-country'] = self.format_native_country(data['native-country'])
        data['income'] = self.format_income(data['income'])
        return data
    '''
        Functions for altering column data
    '''
    def format_age(self, column):
        return self.normalize(column)

    def format_workclass(self, column):
        if self.workclass_categories is not None:
            return self.apply_category(column, self.workclass_categories)
        else:
            result = self.index_string(column)
            self.workclass_categories = result[1]
            return result[0]

    def format_fnlwgt(self, column):
        return self.normalize(column)

    def format_education(self, column):
        if self.education_categories is not None:
            return self.apply_category(column, self.education_categories)
        else:
            result = self.index_string(column)
            self.education_categories = result[1]
            return result[0]

    def format_education_num(self, column):
        return self.normalize(column)

    def format_marital_status(self, column):
        if self.marital_status_categories is not None:
            return self.apply_category(column, self.marital_status_categories)
        else:
            result = self.index_string(column)
            self.marital_status_categories = result[1]
            return result[0]

    def format_occupation(self, column):
        if self.occupation_categories is not None:
            return self.apply_category(column, self.occupation_categories)
        else:
            result = self.index_string(column)
            self.occupation_categories = result[1]
            return result[0]

    def format_relationship(self, column):
        if self.relationship_categories is not None:
            return self.apply_category(column, self.relationship_categories)
        else:
            result = self.index_string(column)
            self.relationship_categories = result[1]
            return result[0]

    def format_race(self, column):
        if self.race_categories is not None:
            return self.apply_category(column, self.race_categories)
        else:
            result = self.index_string(column)
            self.race_categories = result[1]
            return result[0]

    def format_sex(self, column):
        if self.sex_categories is not None:
            return self.apply_category(column, self.sex_categories)
        else:
            result = self.index_string(column)
            self.sex_categories = result[1]
            return result[0]

    def format_capital_gain(self, column):
        return self.normalize(column)

    def format_capital_loss(self, column):
        return self.normalize(column)

    def format_hours_per_week(self, column):
        return self.normalize(column)

    def format_native_country(self, column):
        if self.native_country_categories is not None:
            return self.apply_category(column, self.native_country_categories)
        else:
            result = self.index_string(column)
            self.native_country_categories = result[1]
            return result[0]

    def format_income(self, column):
        if self.income_categories is not None:
            return self.apply_category(column, self.income_categories)
        else:
            result = self.index_string(column)
            self.income_categories = result[1]
            return result[0]

    '''
        Helper functions
    '''
    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def index_string(self, x):
        return pd.factorize(x)

    def apply_category(self, data, category):
        return category.get_indexer(data)

    '''
        Retrieve input features as numpy array
    '''
    def get_data(self, type="test", index=0, batch=0):

        if type == "train":
            if ((index * batch) + batch >= len(self.train)):
                return None
            if batch:
                result = self.train[index*batch:(index*batch)+batch]
                x = result.drop('income', axis=1)
                y = result["income"]
                return np.array(x), np.array(y)
            else:
                result = self.train
                x = result.drop('income', axis=1)
                y = result["income"]
                return np.array(x), np.array(y)
        elif type == "test":
            if ((index * batch) + batch >= len(self.test)):
                return None
            if batch:
                result = self.test[index*batch:(index*batch)+batch]
                x = result.drop('income', axis=1)
                y = result["income"]
                return np.array(x), np.array(y)
            else:
                result = self.test
                x = result.drop('income', axis=1)
                y = result["income"]
                return np.array(x), np.array(y)
