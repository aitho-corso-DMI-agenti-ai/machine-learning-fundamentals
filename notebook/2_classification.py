import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Classification
    return


@app.cell
def _(mo):
    mo.md(r"""# Classification""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection
    from sklearn.model_selection import StratifiedKFold
    return (
        LogisticRegression,
        StratifiedKFold,
        matplotlib,
        model_selection,
        np,
        pd,
        plt,
    )


@app.cell
def _(pd):
    train_df = pd.read_csv("data/train_preprocessed.csv")
    test_df = pd.read_csv("data/test_preprocessed.csv")


    train_df.head()
    return test_df, train_df


@app.cell
def _(test_df, train_df):
    # split features and target variable

    target_name = "fraud_reported"
    features = train_df.columns[train_df.columns!=target_name]


    y_train = train_df[target_name] 
    X_train = train_df[features]

    y_test = test_df[target_name]
    X_test = test_df[features]
    return X_test, X_train, features, target_name, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Simple Logistic regression

        \[
        P(y=1 \mid \mathbf{x}) = \sigma(z) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}
        \]

        """
    )
    return


@app.cell
def _(np, plt):
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Genera dati per z
    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(z, s, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', color='blue')
    plt.axvline(0, color='gray', linestyle='--')
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.title("Sigmoid Function (Logistic Regression)")
    plt.xlabel("z")
    plt.ylabel(r"$\sigma(z)$")
    plt.legend()
    plt.grid(True)
    plt.show()
    return s, sigmoid, z


@app.cell
def _(
    LogisticRegression,
    StratifiedKFold,
    X_train,
    features,
    logistic,
    model_selection,
    np,
    plt,
    y_train,
):
    simple_logistic = LogisticRegression(random_state=123)
    simple_logistic.fit(X_train, y_train)

    slogistic_eval = model_selection.cross_val_score(simple_logistic, X_train, y_train, cv=StratifiedKFold(n_splits=10, random_state=1234, shuffle=True))

    print (f"Average accuracy = {np.average(slogistic_eval):3.2f} +/- {np.std(slogistic_eval): 3.2f}")

    plt.bar(np.arange(len(features)), logistic.coef_[0], color='green',width=0.25)

    return simple_logistic, slogistic_eval


app._unparsable_cell(
    r"""
    Questions

    * Are the performance 'good'?
    * Can the parameters be optimized?
    * Are there other better performing models?
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md(r"""## Hyper parameter tuning""")
    return


@app.cell
def _():
    from sklearn.model_selection import GridSearchCV

    return (GridSearchCV,)


@app.cell
def _(GridSearchCV, LogisticRegression, StratifiedKFold, X_train, y_train):
    param_grid = {
        'C': [0.01, 1],                       # Reguralizzation
        'penalty': ['l2', 'l1'],              # L1/L2 (solo 'l2' se solver='lbfgs')
        'class_weight': ['balanced', None]    # Add class weight

    }

    log_reg = LogisticRegression(max_iter=1000)
    cv_strategy = StratifiedKFold(n_splits=5, random_state=52725, shuffle=True)


    # 6. Grid Search con cross-validation
    grid_search = GridSearchCV(log_reg, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)


    print("Best params found:", grid_search.best_params_)
    print("Best model:", grid_search.best_estimator_)

    optimized_logistic = grid_search.best_estimator_
    return cv_strategy, grid_search, log_reg, optimized_logistic, param_grid


@app.cell
def _(grid_search, np):
    best_idx = np.nanargmax(grid_search.cv_results_['mean_test_score'])

    mean_score = grid_search.cv_results_["mean_test_score"][best_idx]
    std_score = grid_search.cv_results_["std_test_score"][best_idx]

    print(f"Average accuracy = {mean_score:3.2f} +/- {std_score: 3.2f}")
    return best_idx, mean_score, std_score


@app.cell
def _(grid_search):
    grid_search.cv_results_



    return


@app.cell
def _(
    StratifiedKFold,
    X_train,
    model_selection,
    np,
    optimized_logistic,
    y_train,
):
    optimized_logistic_eval = model_selection.cross_val_score(optimized_logistic, X_train, y_train, cv=StratifiedKFold(n_splits=10, random_state=1234, shuffle=True))

    print (f"Average accuracy = {np.average(optimized_logistic):3.2f} +/- {np.std(optimized_logistic): 3.2f}")

    return (optimized_logistic_eval,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Comparing models

        Although all the models have been evaluated have been evaluated with cross-validation, we don't know whether the folds are the same. Thus for instance we don't know if the first element in the evaluation array was measured on the same fold over all the algorithms. Accordingly, we need to apply **unpaired t-Test**.

        Let's compare the performance of Logistic Regression vs Random Forests.
        """
    )
    return


@app.cell
def _():
    from scipy import stats
    return (stats,)


@app.cell
def _(
    RandomForestClassifier,
    StratifiedKFold,
    X_train,
    model_selection,
    np,
    y_train,
):
    rf = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=0)
    rf_eval = model_selection.cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(n_splits=10,random_state=52725,shuffle=True))

    print(f"Random Forest :{np.average(rf_eval): 4.3f} +/- {np.std(rf_eval): 4.3f}")
    return rf, rf_eval


@app.cell
def _(rf_eval, slogistic_eval, stats):
    def PrintSignificance(stat, alpha):
        if (stat[1]<alpha):
            print(f"The difference is statistically significant (α {alpha:3.2f})")
        else:
            print(f"The difference is not statistically significant (α={alpha:3.2f})")
        
    alpha = 0.05

    unpaired_lr_rf = stats.ttest_ind(slogistic_eval, rf_eval)
    print("Logistic Regression vs Random Forests: p-value = %4.3f"%unpaired_lr_rf[1])
    PrintSignificance(unpaired_lr_rf, alpha)
    return PrintSignificance, alpha, unpaired_lr_rf


@app.cell
def _(mo):
    mo.md(
        """
        # Evaluation of the final model

        ## Confusion matrix
        """
    )
    return


@app.cell
def _():
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    return (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(X_test, logistic):
    y_pred = logistic.predict(X_test)
    return (y_pred,)


@app.cell
def _(
    X_test,
    confusion_matrix,
    f1_score,
    matplotlib,
    np,
    plt,
    precision_score,
    recall_score,
    roc_auc_score,
    y_pred,
    y_test,
):
    def plot_confusion_matrix(y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        font = {'size'   : 18}

        matplotlib.rc('font', **font)
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
 
        fig, ax = plt.subplots(figsize=(10,7))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        cm_label= {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP" }
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{format(cm[i, j], fmt)} \n ({cm_label[(i,j)]})",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    print("Number of observation:",len(X_test))
    print("True Label:", y_test.sum())
    print(f"Ratio between number of frauds and total observations: {round(100*y_test.mean(),3)}%")
    print("Predicted frauds:",y_pred.sum())


    print(f"Precision: {round(100*precision_score(y_test,y_pred),3)}%")
    print(f"Recall: {round(100*recall_score(y_test,y_pred),3)}%")
    print(f"F1: {round(100*f1_score(y_test,y_pred),3)}%")
    print(f"Roc-AUC: {round(100*roc_auc_score(y_test,y_pred),3)}%")

    plot_confusion_matrix(y_test, y_pred, classes=['Not Fraud',  'Fraud'], title='Confusion matrix')
    return (plot_confusion_matrix,)


app._unparsable_cell(
    r"""
    N.B the performance are lower than the training ones
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
