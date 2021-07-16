def Xgboost_model(x_train, y_train, x_test, y_test):
    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgb_valid = xgb.DMatrix(x_valid, label=y_valid)
    xgb_test = xgb.DMatrix(x_test)
    params = {'booster': 'gbtree',
            'objective': 'rank:pairwise',
            'gamma': 0.5,
            'max_depth': 15, 
            'subsample': 0.5,
            'colsample_bytree': 0.5,
            'min_child_weight': 1,
            'scale_pos_weight': 1,
            'silent': 0,
            'eta': 0.015,
            'nthread': 8,
            'eval_metric': 'logloss'
            }
    plst = list(params.items())
    num_rounds = 500
    watchlist = [(xgb_valid, 'valid')]
    model = xgb.train(plst, xgb_train, num_rounds, 
watchlist, early_stopping_rounds=200)
    pred_train = model.predict(xgb_train)
    pred_test = model.predict(xgb_test)
    result = [roc_auc_score(y_train,pred_train), roc_auc_score(y_test,pred_test)]
    return model, result, (y_test.values, pred_test)
