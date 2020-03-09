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
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GridSearchCV
    
    # Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc= MinMaxScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    # Training SVM
    from sklearn.svm import SVC
    svc_clf=SVC(kernel ='linear', C=1, gamma=1)
    svc_clf.fit(x_train, y_train)
    y_pred_svc= svc_clf.predict(x_test)
    print(accuracy_score(y_test, y_pred_svc))
    
    cm_svc = confusion_matrix(y_test, y_pred_svc)
    print(cm_svc)
    
    # confusion matrix - SVM
    labels = ["A","B","C","D","E","F"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_svc)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Grid Search - SVM
    parameters = {'kernel':['linear'], 'C':[0.00001, 0.0001,0.001,0.01,0.1, 0.5, 0.8]}
    # parameters = {'kernel':['linear'], 'C':[0.1,10,20,40,70,100,200,260,512]}
    clfGridSV = GridSearchCV(clf1,parameters,cv=3)
    clfGridSV.fit(x_train,y_train)
    accuracy_SVM=clfGridSV.cv_results_['mean_test_score']
    plt.ylabel('Accuracy Of Linear Kernel SVM')
    plt.xlabel('Regularization Parameter (C)')
    plt.plot([0.00001, 0.0001,0.001,0.01,0.1, 0.5, 0.8],accuracy_SVM)
    
    # Training KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    y_pred_knn = knn_model.predict(x_test)

    print(accuracy_score(y_test, y_pred_knn))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    print(cm_knn)
    
    # Confusion Matrix - KNN
    labels = ["A","B","C","D","E","F"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_knn)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Grid Search - KNN
    parameters = {'n_neighbors': [1,3,5,10,15,30]}
    gs_knn = GridSearchCV(model,parameters,cv=5)
    gs_knn.fit(x_train,y_train)
    accuracy_gs_knn=gs_knn.cv_results_['mean_test_score']
    plt.ylabel('Accuracy Of KNN')
    plt.xlabel('Regularization Parameter (n_neighbor)')
    plt.plot([1,3,5,10,15,30],accuracy_gs_knn)
    
    
    
    

def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """


"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""



    
