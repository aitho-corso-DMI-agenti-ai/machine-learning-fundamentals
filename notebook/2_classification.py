import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    # Ignore ConvergenceWarning
    warnings.filterwarnings("ignore", category = ConvergenceWarning)
    return (
        ConvergenceWarning,
        LogisticRegression,
        StratifiedKFold,
        matplotlib,
        model_selection,
        np,
        pd,
        plt,
        warnings,
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
def _(LogisticRegression, X_train, y_train):
    simple_logistic = LogisticRegression(random_state=12345)
    simple_logistic.fit(X_train, y_train)
    simple_logistic
    return (simple_logistic,)


@app.cell
def _(
    StratifiedKFold,
    X_train,
    features,
    model_selection,
    np,
    plt,
    simple_logistic,
    y_train,
):
    slogistic_eval = model_selection.cross_val_score(simple_logistic, X_train, y_train, cv=StratifiedKFold(n_splits=10, random_state=1234, shuffle=True))


    print(f"Average accuracy = {np.average(slogistic_eval):3.2f} +/- {np.std(slogistic_eval): 3.2f}")
    plt.bar(np.arange(len(features)), simple_logistic.coef_[0], color='green',width=0.25)
    return (slogistic_eval,)


@app.cell
def _(mo):
    mo.md(
        """
        Questions

        * Are the performance good?
        * Can the parameters be optimized?
        * Are there other better performing models?
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Hyper parameter tuning""")
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
        'solver': ['liblinear']               # Algorithm to use in the optimization problem
    }

    log_reg = LogisticRegression(max_iter=1000)
    cv_strategy = StratifiedKFold(n_splits=10, random_state=52725, shuffle=True)


    # 6. Grid Search con cross-validation
    grid_search = GridSearchCV(
        log_reg, 
        param_grid,
        cv=cv_strategy, 
        scoring='accuracy', 
        verbose=1
    )
    grid_search.fit(X_train, y_train)


    print("Best params found:", grid_search.best_params_)
    print("Best model:", grid_search.best_estimator_)

    optimized_logistic = grid_search.best_estimator_
    optimized_logistic
    return cv_strategy, grid_search, log_reg, optimized_logistic, param_grid


@app.cell
def _(grid_search, np):
    best_idx = grid_search.best_index_

    optimized_logistic_eval = np.array([grid_search.cv_results_[f"split{i}_test_score"][best_idx] for i in range(10)])

    mean_score = grid_search.cv_results_["mean_test_score"][best_idx]
    std_score = grid_search.cv_results_["std_test_score"][best_idx]

    print(f"Average accuracy = {mean_score:3.2f} +/- {std_score: 3.2f}")
    return best_idx, mean_score, optimized_logistic_eval, std_score


@app.cell
def _(mo):
    mo.md("""## Multi Layer Perceptron""")
    return


@app.cell
def _():
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    return Adam, Dense, EarlyStopping, Input, Sequential


@app.cell
def _(Adam, Dense, Input, Sequential, X_train):
    input_dim = X_train.shape[1]

    def build_model():
    # Costruzione del modello

        model_ = Sequential([
            Input(shape=(input_dim,)),        # Primo layer di input
            Dense(16, activation='relu'),     # Hidden layer con 16 neuroni
            Dense(1, activation='sigmoid')    # Output layer per classificazione binaria
        ])

        model_.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model_

    model = build_model()
    model.summary()
    return build_model, input_dim, model


@app.cell
def _(EarlyStopping, X_train, model, y_train):
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,                 # number of epochs to wait before stopping
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    return early_stop, history


@app.cell
def _(history):
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"Accuracy finale sul training set: {final_train_acc:.4f}")
    print(f"Accuracy finale sul validation set: {final_val_acc:.4f}")
    return final_train_acc, final_val_acc


@app.cell
def _(history, plt):
    # Tracciamento della curva della perdita
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss in time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Tracciamento della curva dell'accuratezza
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy in time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    # Cross validation for estimating performances
    return


@app.cell
def _(
    StratifiedKFold,
    X_train,
    accuracy_score,
    build_model,
    early_stop,
    np,
    y_train,
):
    def cv_scores():
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
        keras_scores = []

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = build_model()
            model.fit(
                X_tr, 
                y_tr,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
            )

            y_pred = (model.predict(X_val) > 0.5).astype("int32")
            acc = accuracy_score(y_val, y_pred)
            keras_scores.append(acc)

        keras_scores = np.array(keras_scores)
        return keras_scores

    keras_scores = cv_scores()
    print("Cross-validated accuracies del modello Keras:", keras_scores)
    print("Media accuracy:", keras_scores.mean())

    return cv_scores, keras_scores


@app.cell
def _(mo):
    mo.md(
        r"""
        # Comparing models

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
def _():
    def PrintSignificance(stat, alpha):
        if (stat[1]<alpha):
            print(f"The difference is statistically significant (α {alpha:3.2f})")
        else:
            print(f"The difference is not statistically significant (α={alpha:3.2f})")

    alpha = 0.01

    return PrintSignificance, alpha


@app.cell
def _(
    PrintSignificance,
    alpha,
    optimized_logistic_eval,
    slogistic_eval,
    stats,
):
    unpaired_slr_olr = stats.ttest_ind(slogistic_eval, optimized_logistic_eval)
    print("\033[1m Simple Logistic Regression vs Optimezed Logistic Regression \033[0m ")
    print("p-value = %4.3f"%unpaired_slr_olr[1])
    PrintSignificance(unpaired_slr_olr, alpha)
    return (unpaired_slr_olr,)


@app.cell
def _(PrintSignificance, alpha, keras_scores, slogistic_eval, stats):
    unpaired_slr_mlp= stats.ttest_ind(slogistic_eval, keras_scores)
    print("\033[1m Simple Logistic Regression vs Multi layer perceptron \033[0m ")
    print("p-value = %4.3f"%unpaired_slr_mlp[1])
    PrintSignificance(unpaired_slr_mlp, alpha)
    return (unpaired_slr_mlp,)


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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    return (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )


@app.cell
def _(X_test, simple_logistic):
    y_pred = simple_logistic.predict(X_test)
    return (y_pred,)


@app.cell
def _(
    X_test,
    accuracy_score,
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
    print(f"Predicted frauds: {y_pred.sum()}\n")


    print(f"Accuracy: {round(100*accuracy_score(y_test,y_pred),3)}%")
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
