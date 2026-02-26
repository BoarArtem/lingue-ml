import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


class B2PredictorModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.feature_names: list[str] = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.feature_names = X_train.columns.tolist()
        self.model.fit(X_train, y_train)

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df[self.feature_names])

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        preds = self.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"MAE: +/- {mae:.1f} дней")
        print(f"R2: {r2:.2f}")

    def show_feature_importance(self):
        if not self.feature_names:
            return

        importances = self.model.feature_importances_

        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        print(feature_imp.head(5))