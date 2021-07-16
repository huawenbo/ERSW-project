def Lightgbm_model(x_train, y_train, x_test, y_test):
    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_valid = lgb.Dataset(x_valid, label=y_valid)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 30,
        'max_depth': 15,
        'max_bin': 50,
        'min_data_in_leaf': 15,
        'learning_rate': 0.015,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'scale_pos_weight': 50
    }
MAX_ROUNDS = 500
    model = lgb.train(params, lgb_train, valid_sets=lgb_valid, 
                  num_boost_round=MAX_ROUNDS,
early_stopping_rounds=200)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    result = [roc_auc_score(y_train,pred_train), roc_auc_score(y_test,pred_test)]
    return model, result, (y_test.values, pred_test)
