import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from utils.common import validation



def svm_baseline(X, Y, X_test, Y_test, method=None):
    clf = SVC(gamma='auto', class_weight='balanced', probability=True).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('SVM baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))
    if (method != None):
        with open('./reports/results/{}_SVM.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_svm, tpr_rt_svm, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_svm, tpr_rt_svm))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_svm, tpr_rt_svm, label='SVM')
    plt.legend(loc='best')


def random_forest_baseline(X, Y, X_test, Y_test, method=None):
    clf = RandomForestClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('Rrandom Forest baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))
    if (method != None):
        with open('./reports/results/{}_RF.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_rf, tpr_rt_rf, _ = roc_curve(Y_test, y_pred_roc)
    plt.figure(1)
    print(auc(fpr_rt_rf, tpr_rt_rf))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_rf, tpr_rt_rf, label='RF')
    plt.legend(loc='best')


def knn_baseline(X, Y, X_test, Y_test, method=None):
    clf = KNeighborsClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('knn baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))
    if (method != None):
        with open('./reports/results/{}_SVM.txt'.format(method), 'a') as f:
            f.write(' T_Accuracy:\t%.3f\n' % train_acc)
            f.write(' T_Precision:\t%.3f\n' % train_pre)
            f.write(' T_Recall:\t%.3f\n' % train_rec)
            f.write(' T_F1-score:\t%.3f\n' % train_fscore)
            f.write(' T_Matthews CC:\t%.3f\n\n' % train_mcc)
            f.write(' V_Accuracy:\t%.3f\n' % val_acc)
            f.write(' V_Precision:\t%.3f\n' % precision)
            f.write(' V_Recall:\t%.3f\n' % recall)
            f.write(' V_F1-score:\t%.3f\n' % fscore)
            f.write(' V_Matthews CC:\t%.3f\n\n' % mcc)

    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_knn, tpr_rt_knn, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_knn, tpr_rt_knn))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_knn, tpr_rt_knn, label='LR')
    plt.legend(loc='best')
    plt.show()


def bayes_baseline(X, Y, X_test, Y_test, method=None):
    clf = GaussianNB().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('bayes baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))

    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_nb, tpr_rt_nb, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_nb, tpr_rt_nb))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_nb, tpr_rt_nb, label='LR')
    plt.legend(loc='best')
    plt.show()


def logistic_regression_baseline(X, Y, X_test, Y_test, method=None):
    clf = LogisticRegression(random_state=0).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = validation.evaluate(Y_test, Y_pred)
    print('Logistic regression baseline:')
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))
    # roc curve
    y_pred_roc = clf.predict_proba(X_test)[:, 1]
    fpr_rt_lr, tpr_rt_lr, _ = roc_curve(Y_test, y_pred_roc)
    print(auc(fpr_rt_lr, tpr_rt_lr))
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lr, tpr_rt_lr, label='SVM')
    plt.legend(loc='best')
    # plt.show()