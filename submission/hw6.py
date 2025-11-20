#!/bin/python3
"""Scott Andersen - ATMS 523 - HW6."""
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr


def aggregate_tornado_data(filename="1950-2021_actual_tornadoes.csv"):
    """Agregate the monthly tornado data."""
    data = pd.read_csv(filename)
    # select illinois data
    ill = data[data.st == 'IL']
    # group by the year and month, and count
    t_count = ill[['yr', 'mo', 'om']].groupby(['yr', 'mo']).aggregate('count')
    t_count = t_count.reset_index()
    t_count['day'] = 1
    # create a date index that matches the oscillation data format
    t_count['date'] = pd.to_datetime(
            t_count.rename({
                'yr': 'year',
                'mo': 'month'},
                axis=1)[['year', 'month', 'day']]
            )
    # clean up the dataframe
    t_count = t_count.drop(['yr', 'mo', 'day'], axis=1)
    t_count = t_count.rename({'om': 'count'}, axis=1)
    t_count = t_count.set_index('date')
    return t_count


def load_oscillation_data(filename="oscill.csv"):
    """Load the oscillation data."""
    osc = pd.read_csv(filename)
    # match dateformat to the tornado data
    osc = osc.rename({'Date': 'date'}, axis=1)
    osc['date'] = pd.to_datetime(osc['date'])
    return osc.set_index('date')


def merge_data(tornado, oscillation):
    """Merge the data and join on the date index."""
    # inner join on the date
    return oscillation.merge(tornado, left_on='date',
                             right_on='date', how='inner')


def prepare_and_split_data(df, seed=251116):
    """Prep the data and return X and Y  sets."""
    x = df.drop('count', axis=1).to_numpy()
    y = df['count'].to_numpy()

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=seed)

    return x_train, x_test, y_train, y_test


def random_forest_predict(x_train, y_train, x_test):
    """Predict using the random forest regressor."""
    param_grid = {
         "bootstrap": [True, False],
         "max_depth": [10, 50],
         "max_features": ["sqrt", 1.0],
         "min_samples_leaf": [1, 4],
         "min_samples_split": [2, 10],
         "n_estimators": [200, 500]
     }

    forest_model = RandomForestRegressor()

    grid = GridSearchCV(forest_model, param_grid, cv=7, n_jobs=-1)

    # fit the model and get the best estimator
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    return model.predict(x_test)


def evaluate(y_pred, y_true, print_scores=True):
    """Calculate the evaluation scores."""
    scores = {
        "pearson_coef": pearsonr(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }

    if print_scores:
        print(f"Coef:  {scores['pearson_coef'][0]:.4f} "
              f"(p-val {scores['pearson_coef'][1]:.4f})")
        print(f"RMSE: {scores['RMSE']:.4f}")

    return scores


def scatter_plot(y_pred, y_true, title=""):
    """Create a scatter plot with the true and predicted data."""
    plt.scatter(y_pred, y_true)
    plt.xlabel("Predicted Tornado Counts")
    plt.ylabel("True Tornado Counts")
    plt.title(title)

    plt.show()


def one_hot_data(df):
    """Create a one hot encoding for the data."""
    df['month'] = df.index.month
    return pd.get_dummies(df, columns=['month'], dtype=np.int8)


def feature_importance(df, x_train, x_test, y_train, y_test):
    """Calculate the feature importance."""
    param_grid = {
         "bootstrap": [True, False],
         "max_depth": [10, 50],
         "max_features": ["sqrt", 1.0],
         "min_samples_leaf": [1, 4],
         "min_samples_split": [2, 10],
         "n_estimators": [200, 500]
     }

    forest_model = RandomForestRegressor()

    grid = GridSearchCV(forest_model, param_grid, cv=7, n_jobs=-1)

    # fit the model and get the best estimator
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(x_test)
    evaluate(y_pred, y_test)

    # perform the analysis
    result = permutation_importance(
        model, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=df.drop(['count'], axis=1).columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.show()


def shap_analysis(df, x_train, x_test, y_train, y_test):
    """Perform the shap analysis."""
    param_grid = {
         "bootstrap": [True, False],
         "max_depth": [10, 50],
         "max_features": ["sqrt", 1.0],
         "min_samples_leaf": [1, 4],
         "min_samples_split": [2, 10],
         "n_estimators": [200, 500]
     }

    forest_model = RandomForestRegressor()

    grid = GridSearchCV(forest_model, param_grid, cv=7, n_jobs=-1)

    # fit the model and get the best estimator
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    shap.initjs()

    x_train_df = pd.DataFrame(x_train, columns=df.drop(['count'],
                              axis=1).columns)
    x_test_df = pd.DataFrame(x_test,  columns=df.drop(['count'],
                             axis=1).columns)

    explainer = shap.Explainer(model, x_train_df, feature_names=df.columns)
    shap_values = explainer(x_test_df, check_additivity=False)
    shap.plots.beeswarm(shap_values, max_display=16)
    shap.plots.bar(shap_values, max_display=16)

    # zeroth index is the ENSO
    shap.plots.scatter(shap_values[:, 'ENSO'], color=shap_values)


def main():
    """Execute the assignment pipeline."""
    # aggregate tornado data and load the oscillation data from module 4
    tornado = aggregate_tornado_data()
    oscillation = load_oscillation_data()
    data = merge_data(tornado, oscillation)

    # prepare the data split
    x_train, x_test, y_train, y_test = prepare_and_split_data(data)

    print("Random forest prediction")
    # basic random forest prediction and the scatter plot
    y_pred_forest = random_forest_predict(x_train, y_train, x_test)
    evaluate(y_pred_forest, y_test)
    scatter_plot(y_pred_forest, y_test, "Random Forest Tornado Prediction")

    # creae the one hot encoded months and create the train test split
    hot_data = one_hot_data(data)
    hot_data.to_csv("tornados.csv")
    x_train, x_test, y_train, y_test = prepare_and_split_data(hot_data)

    print("One hot Random forest prediction")
    # repeat the random forest prediction
    y_pred_forest = random_forest_predict(x_train, y_train, x_test)
    evaluate(y_pred_forest, y_test)
    scatter_plot(y_pred_forest, y_test,
                 "One-Hot Random Forest Tornado Prediction")

    # feature importance
    print("Feature importance analysis")
    feature_importance(hot_data, x_train, x_test, y_train, y_test)

    # shap
    print("Shap analysis")
    shap_analysis(hot_data, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
