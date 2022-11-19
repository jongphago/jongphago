import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib


class RandomForestRegressor(RandomForestRegressor):
    def __init__(
            self,
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
        self.grid_search_ = None
        self.attributes = None

    @staticmethod
    def root_mean_squared_error(predictions, labels):
        mse = mean_squared_error(predictions, labels)
        rmse = np.sqrt(mse)
        return rmse

    def cross_val_score_with_rmse(self, prepared, labels):
        _scores = cross_val_score(self, prepared, labels,
                                  scoring="neg_mean_squared_error",
                                  cv=2,
                                  verbose=2)
        return np.sqrt(-_scores)

    def grid_search_cv(self, param_grid, prepared, labels):
        self.grid_search_ = GridSearchCV(self, param_grid, cv=2,
                                         scoring='neg_mean_squared_error',
                                         return_train_score=True,
                                         verbose=2
                                         )
        self.grid_search_.fit(prepared, labels)

    @property
    def importances(self):
        return self.grid_search_.best_estimator_.feature_importances_

    # TODO
    def set_attributes(self, num_attribs, extra_attribs, cat_one_hot_attribs):
        self.attributes = num_attribs + extra_attribs + cat_one_hot_attribs

    def save_model(self, file_name: str = 'my_model.pkl'):
        joblib.dump(self, file_name)
