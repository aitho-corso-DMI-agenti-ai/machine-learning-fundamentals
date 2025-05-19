import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # **Insurance claims**

        The aim of this notebook is to prepare the *Insurance claims* dataset to be ingested by the the model. 
        The dataset could be found on Kaggle searching for [Insurance claims dataset](https://www.kaggle.com/code/buntyshah/insurance-fraud-claims-detection).

        <img src="img/Slide4.png" width="750">
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Pre-processing""")
    return


@app.cell
def _():
    # libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""## Data Analysis""")
    return


@app.cell
def _(pd):
    data_filename = r"data\insurance_claims.csv"
    raw_df = pd.read_csv(data_filename)
    raw_df.head()
    return data_filename, raw_df


@app.cell
def _(pd, raw_df):
    date_cols = ["policy_bind_date", "incident_date"]

    df = raw_df.drop('_c39', axis=1)
    df['days_start_incident'] = (pd.to_datetime(df['incident_date']) - pd.to_datetime(df['policy_bind_date'])).dt.days

    df.head()
    return date_cols, df


@app.cell
def _(raw_df):
    print('Dimensions of the dataset:', raw_df.shape)
    raw_df.describe(include='all')

    return


@app.cell
def _(mo):
    mo.md("""## Missing columns""")
    return


@app.cell
def _(df, pd):
    # Checking missing values
    # Function to calculate missing values by column# Funct 
    def missing_values_table(df):
            # Total missing values
            mis_val = df.isnull().sum()

            # Percentage of missing values
            mis_val_percent = 100 * df.isnull().sum() / len(df)

            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values'})

            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")

            # Return the dataframe with missing information
            return mis_val_table_ren_columns

    # Missing values statistics
    missing_values = missing_values_table(df)
    missing_values
    return missing_values, missing_values_table


@app.cell
def _(df):
    df['authorities_contacted'].value_counts(dropna=False)
    return


@app.cell
def _(df):
    # numerical: discrete
    discrete_cols = [
        var for var in df.columns if df[var].dtype != 'O' and var not in ['fraud_reported']
        and df[var].nunique() < 10
    ]

    df[discrete_cols].head()
    return (discrete_cols,)


@app.cell
def _(df, discrete_cols):

    # numerical: continuous
    continuous_cols = [
        var for var in df.columns
        if df[var].dtype != 'O' and var != 'fraud_reported' and var not in discrete_cols
    ]

    df[continuous_cols].head()
    return (continuous_cols,)


@app.cell
def _(date_cols, df):
    categorical_cols = [var for var in df.columns if df[var].dtype == 'O' and var != 'fraud_reported' and var not in date_cols]
    df[categorical_cols].head()
    return (categorical_cols,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Plots

        ### Heatmap
        """
    )
    return


@app.cell
def _(continuous_cols, df, discrete_cols, plt, sns):
    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(df[discrete_cols + continuous_cols].corr(), cmap=cmap, vmax=.3, center=0,annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax, cmap, f


@app.cell
def _():
    ## Unique columns
    return


@app.cell
def _(df, pd):
    colum_name =[]
    unique_value=[]
    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            colum_name.append(str(col)) 
            unique_value.append(df[col].nunique())
    table= pd.DataFrame()
    table['Col_name'] = colum_name
    table['Value']= unique_value

    table=table.sort_values('Value',ascending=False)
    table
    return col, colum_name, table, unique_value


@app.cell
def _(df, table):
    df[table.Col_name[:4]].head()
    return


@app.cell
def _(mo):
    mo.md(r"""## Target variable""")
    return


@app.cell
def _(df, sns):
    label_col = "fraud_reported"

    val_count = (df[label_col].value_counts() / len(df[label_col])*100).round(1)

    print(f"Value distribution in percentage: \n {val_count}")
    print('-'*30)

    sns.countplot(x=df['fraud_reported'])

    return label_col, val_count


@app.cell
def _(mo):
    mo.md(
        r"""
        # Feature Engineering

        ## Datetime features
        """
    )
    return


@app.cell
def _():
    ## Train test Split
    return


@app.cell
def _(df):
    from sklearn.model_selection import train_test_split


    y = df['fraud_reported'].map({'Y': True, 'N': False})
    X = df.drop('fraud_reported', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X, X_test, X_train, train_test_split, y, y_test, y_train


@app.cell
def _(mo):
    mo.md("""## Set up the Pipeline""")
    return


@app.cell
def _(pd):
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin

    class DateFeatureExtractor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            self.output_features_ = []
            for col in X.columns:
                self.output_features_.extend([
                    f"{col}_anno",
                    f"{col}_mese",
                    f"{col}_giorno_settimana"
                ])
            return self

        def transform(self, X):
            df = X.copy()
            for col in df.columns:
                parsed = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_anno'] = parsed.dt.year
                df[f'{col}_mese'] = parsed.dt.month
                df[f'{col}_giorno_settimana'] = parsed.dt.dayofweek
            return df[self.output_features_]

        def get_feature_names_out(self, input_features=None):
            return np.array(self.output_features_)

    return BaseEstimator, DateFeatureExtractor, TransformerMixin, np


@app.cell
def _(
    DateFeatureExtractor,
    categorical_cols,
    continuous_cols,
    date_cols,
    discrete_cols,
):
    from sklearn.compose import ColumnTransformer

    from sklearn.preprocessing import StandardScaler,  FunctionTransformer, OneHotEncoder


    preprocessor = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05, max_categories=5), categorical_cols),
            ("scale",  StandardScaler(), continuous_cols),
            ("identity", "passthrough", discrete_cols),
            ("dates",  DateFeatureExtractor(), date_cols)
            ], 
        remainder='drop',
        n_jobs=-1
    )

    preprocessor
    return (
        ColumnTransformer,
        FunctionTransformer,
        OneHotEncoder,
        StandardScaler,
        preprocessor,
    )


@app.cell
def _(X_test, X_train, pd, preprocessor):
    # Applico il preprocessor al dataset

    X_train_prep = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    X_test_prep = preprocessor.transform(X_test)


    X_train_prep_df = pd.DataFrame(X_train_prep, columns=feature_names, index=X_train.index)
    X_test_prep_df = pd.DataFrame(X_test_prep, columns=feature_names, index=X_test.index)


    X_train_prep_df.head()
    return (
        X_test_prep,
        X_test_prep_df,
        X_train_prep,
        X_train_prep_df,
        feature_names,
    )


@app.cell
def _(mo):
    mo.md("""## Save dataset""")
    return


@app.cell
def _(X_test_prep_df, X_train_prep_df, y_test, y_train):
    # Salvataggio unico

    df_train = X_train_prep_df.copy()
    df_train['fraud_reported'] = y_train.values
    df_train.to_csv("data/train_preprocessed.csv", index=False)


    df_test = X_test_prep_df.copy()
    df_test['fraud_reported'] = y_test.values
    df_test.to_csv("data/test_preprocessed.csv", index=False)
    return df_test, df_train


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
