def SVM_model(x_train, y_train, x_test, y_test):
    model = SVC(kernel='rbf', probability=True)  
    model.fit(x_train,y_train)  
    pred_train = model.predict_proba(x_train)[:,-1]
    pred_test = model.predict_proba(x_test)[:,-1]
    result = [roc_auc_score(y_train,pred_train), roc_auc_score(y_test,pred_test)]
    return model, result, (y_test.values, pred_test)
