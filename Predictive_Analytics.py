# -*- coding: utf-8 -*-
"""
Predicitve_Analytics.py
"""


def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """

def Recall(y_true,y_pred):
     """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  

def KNN(X_train,X_test,Y_train, N):
     """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    

def SklearnSupervisedLearning(X_train,Y_train,X_test):
    
    results = []
    
    # Scaling
    sc = sklearn.preprocessing.MinMaxScaler()
    x_train = sc.fit_transform(X_train)
    x_test = sc.transform(X_test)
    
    # Training SVM
    
    svc_clf=SVC(kernel ='linear', C=1, gamma=1)
    svc_clf.fit(x_train, Y_train)
    y_pred_svc= svc_clf.predict(x_test)
    print("SVM Accuracy: " + str(sklearn.metrics.accuracy_score(Y_test, y_pred_svc) * 100))
    
    
    
    # confusion matrix - SVM
    cm_svc = confusion_matrix(Y_test, y_pred_svc) 
    
    
    # Training KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, Y_train)
    y_pred_knn = knn_model.predict(X_test)
    print("KNN Accuracy: " + str(sklearn.metrics.accuracy_score(Y_test, y_pred_knn) * 100))
    
    
    # Confusion Matrix - KNN
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_leaf_nodes=50, random_state=0)
    dt.fit(x_train, Y_train)
    y_pred_tree = dt.predict(x_test)
    print("Decision Tree Accuracy: " + str(sklearn.metrics.accuracy_score(Y_test, y_pred_tree) * 100))
    
    # Confusion Matrix - DTree
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    
    # Logistic Regression Model
    logclf = LogisticRegression(random_state = 0, penalty = 'l1', solver='saga', class_weight='balanced', multi_class='multinomial').fit(x_train, Y_train)
    y_pred_log = logclf.predict(x_test)
    print("Logistic Regression Accuracy: " + str(sklearn.metrics.accuracy_score(Y_test, logclf) * 100))
    
    return results

def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    eclf1 = VotingClassifier(estimators=[('lr', logclf), ('dt', dt), ('knn', knn_model), ('svc', svc_clf)], voting='hard')
    eclf1 = eclf1.fit(X_train, Y_train)
    y_ens = logclf.predict(X_test)
    print("Ensemble Model Accuracy: " + str(sklearn.metrics.accuracy_score(Y_test, y_ens) * 100))


def GridSearchSVM(X_train, Y_train):

    # Scaling
    sc = sklearn.preprocessing.MinMaxScaler()
    x_train = sc.fit_transform(X_train)
    
    
    # Grid Search - SVM
    parameters = {'kernel':['linear'], 'C':[0.00001, 0.0001,0.001,0.01,0.1, 0.5, 0.8]}
    # parameters = {'kernel':['linear'], 'C':[0.1,10,20,40,70,100,200,260,512]}
    clfGridSV = GridSearchCV(svc_clf,parameters,cv=3)
    clfGridSV.fit(X_train,Y_train)
    accuracy_SVM=clfGridSV.cv_results_['mean_test_score']
    plt.ylabel('Accuracy Of Linear Kernel SVM')
    plt.xlabel('Regularization Parameter (C)')
    plt.plot([0.00001, 0.0001,0.001,0.01,0.1, 0.5, 0.8],accuracy_SVM)
   
    
def GridSearchSVM(X_train, Y_train):
    
    # Scaling
    sc = sklearn.preprocessing.MinMaxScaler()
    x_train = sc.fit_transform(X_train)
    
    # Grid Search - DTree
    parameters = {'max_depth': [3,6,9,12]}
    gs_tree = GridSearchCV(estimator,parameters,cv=3)
    gs_tree.fit(x_train,y_train)
    accuracy_gs_tree=gs_tree.cv_results_['mean_test_score']
    plt.ylabel('Accuracy Of Decision Tree')
    plt.xlabel('Regularization Parameter (max_depth)')
    plt.plot([3,6,9,12],accuracy_gs_tree)

def GridSearchKNN(X_train, Y_train):
    
    # Grid Search - KNN
    parameters = {'n_neighbors': [1,3,5,10,15,30]}
    gs_knn = GridSearchCV(model,parameters,cv=5)
    gs_knn.fit(x_train,y_train)
    accuracy_gs_knn=gs_knn.cv_results_['mean_test_score']
    plt.ylabel('Accuracy Of KNN')
    plt.xlabel('Regularization Parameter (n_neighbor)')
    plt.plot([1,3,5,10,15,30],accuracy_gs_knn)



"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""



    
