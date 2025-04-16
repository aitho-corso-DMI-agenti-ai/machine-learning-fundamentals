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
    #libraries
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""## Reading Data""")
    return


@app.cell
def _():
    data_path = r"data"
    data_name = "insurance_claims.csv"
    return data_name, data_path


@app.cell
def _(data_name, data_path, pd):
    raw_df = pd.read_csv(rf"{data_path}\\{data_name}")
    raw_df.head()
    return (raw_df,)


@app.cell
def _(raw_df):
    print('Dimensions of the dataset:', raw_df.shape)
    raw_df.describe(include='all')

    return


@app.cell
def _():
    ## Missing columns
    return


@app.cell
def _(raw_df):
    # remove `_c39` asit is likely an erorr column

    raw_df.drop(columns=['_c39'], inplace=True)
    return


@app.cell
def _(pd, raw_df):
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
    missing_values = missing_values_table(raw_df)
    missing_values
    return missing_values, missing_values_table


@app.cell
def _(raw_df):
    # numerical: discrete
    discrete_cols = [
        var for var in raw_df.columns if raw_df[var].dtype != 'O' and var != 'fraud_reported'
        and raw_df[var].nunique() < 10
    ]

    discrete_cols
    return (discrete_cols,)


@app.cell
def _(discrete_cols, raw_df):

    # numerical: continuous
    continuous_cols = [
        var for var in raw_df.columns
        if raw_df[var].dtype != 'O' and var != 'fraud_reported' and var not in discrete_cols
    ]

    continuous_cols
    return (continuous_cols,)


@app.cell
def _(raw_df):
    categorical_cols = [var for var in raw_df.columns if raw_df[var].dtype == 'O' and var != 'fraud_reported']
    categorical_cols
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
def _(continuous_cols, discrete_cols, plt, raw_df, sns):
    sns.set(style="white")

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 15))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(raw_df[discrete_cols + continuous_cols].corr(), cmap=cmap, vmax=.3, center=0,annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax, cmap, f


@app.cell
def _():
    ## Unique columns
    return


@app.cell
def _(pd, raw_df):
    colum_name =[]
    unique_value=[]
    # Iterate through the columns
    for col in raw_df:
        if raw_df[col].dtype == 'object':
            # If 2 or fewer unique categories
            colum_name.append(str(col)) 
            unique_value.append(raw_df[col].nunique())
    table= pd.DataFrame()
    table['Col_name'] = colum_name
    table['Value']= unique_value
            
    table=table.sort_values('Value',ascending=False)
    table
    return col, colum_name, table, unique_value


@app.cell
def _():
    return


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
def _(raw_df):
    mapping_dict={'Sunday':'Sun','Monday': 'Mon','Tuesday':'Tues',
                  'Wednesday':'Weds','Thursday':'Thurs','Friday':'Fri','Saturday':'Sat'}


    # convert features in datetime objects

    df = raw_df.astype({
        'policy_bind_date': 'datetime64[s]', 
        'incident_date': 'datetime64[s]'}
                   ,)


    df['days_start_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    df['day_of_week'] = df['incident_date'].dt.day_name()#.map(mapping_dict)
    df.head()

    return df, mapping_dict


@app.cell
def _(mo):
    mo.md(r"""# drop to variables cols""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""## Target analysis""")
    return


@app.cell
def _(df, sns):

    def categorical_distribution(df, var_list, min_unique=0):
        for column in var_list:
            print('Number of unique values in {} : {}'.format(column, df[column].nunique()))
            print("Value distribution in percentage of '{0:s}' : \n{1}".format(column,(df[column].value_counts()/len(df[column])*100).round(1)))
            print('-'*30)

    _ = categorical_distribution(df, ['fraud_reported'])


    sns.countplot(x='fraud_reported', data=df)

    return (categorical_distribution,)


@app.cell
def _():
    ## missing data
    return


@app.cell
def _():
    return


@app.cell
def _(df):
    df.drop(['_c39','incident_location', 'insured_zip','policy_number', 
                       'policy_bind_date', 'incident_date'], axis= 1, inplace = True)

    return


@app.cell
def _():
    ## Train test Split
    return


@app.cell
def _(df, train_test_split):

    target_map = {'Y': True, 'N': False}
    y = df['fraud_reported'].map(target_map)

    X = df.drop('fraud_reported', axis=1)

    X_train, t_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # _ = categorical_distribution(train, ['fraud_reported'])
    # _ = categorical_distribution(test, ['fraud_reported'])
    return X, X_test, X_train, t_train, target_map, y, y_test


@app.cell
def _(y):
    y
    return


@app.cell
def _(mo):
    mo.md("""## Set up the Pipeline""")
    return


@app.cell
def _(OneHotEncoder, X_train, column_to_encode, pd, train):
    encoder=OneHotEncoder(handle_unknown='ignore')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder


    train_encoded = pd.DataFrame(encoder.fit_transform(X_train[column_to_encode]))
    train_encoded.columns = encoder.get_feature_names(column_to_encode)
    train.drop(column_to_encode ,axis=1, inplace=True)

    OH_train= pd.concat([train, train_encoded], axis=1)
    return OH_train, OneHotEncoder, encoder, train_encoded, train_test_split


@app.cell
def _(OneHotEncoder, categorical_cols, continuous_cols):
    from sklearn.compose import ColumnTransformer

    from sklearn.preprocessing import StandardScaler


    column_transformer = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05, max_categories=15), categorical_cols),
        ("scale",  StandardScaler(), continuous_cols)
    ], remainder='passthrough', n_jobs=-1
    )

    column_transformer
    return ColumnTransformer, StandardScaler, column_transformer


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
