import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class B2PredictorModel:
    def __init__(self):
       
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.feature_names = []

    def train(self, X_train, y_train):
        
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)
        print("Модель обучена")

    def evaluate(self, X_test, y_test):
        
        preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print("\nЛоги:")
        print(f"MAE(Ошибка в днях): +/- {mae:.1f} дней")
        print(f"Точность: {r2:.2f} процентов")

    def show_feature_importance(self):

        if not self.feature_names:
            return

        importances = self.model.feature_importances_

        feature_imp = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        feature_imp = feature_imp.sort_values(by="Importance", ascending=False)
        print(feature_imp.head(5))

    def get_model(self):
        return self.model