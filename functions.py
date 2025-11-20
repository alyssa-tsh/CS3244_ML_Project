#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub
from kagglehub import KaggleDatasetAdapter
from  sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
import numpy as np
import pandas as pd
from scipy.stats import entropy
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split
from itertools import filterfalse
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def load_data():
  application_records_raw = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rikdifos/credit-card-approval-prediction",
  "application_record.csv",
  )

  credit_records_raw = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rikdifos/credit-card-approval-prediction",
  "credit_record.csv",
  )
  return application_records_raw, credit_records_raw


# In[ ]:


# application record clean functions

def encode_categories(df, col, encoding="label", unknown_label="Unknown", encoders=None):
    
    if encoders is None:
        encoders = {}

    # Ensure column is string and fill missing
    df[col] = df[col].astype(str).fillna(unknown_label)

    if encoding == "none":
        return df, encoders

    # -------------------------
    # Label encoding
    # -------------------------
    if encoding == "label":
        known_mask = df[col] != unknown_label

        if col in encoders:
            le = encoders[col]  # use fitted encoder
        else:
            le = LabelEncoder()
            le.fit(df.loc[known_mask, col])
            encoders[col] = le

        encoded = pd.Series(-1, index=df.index)
        encoded[known_mask] = le.transform(df.loc[known_mask, col])
        df[col + "_encoded"] = encoded.astype(int)

    # -------------------------
    # One-hot encoding
    # -------------------------
    elif encoding == "onehot":
        if col in encoders:
            ohe = encoders[col]
            onehot_arr = ohe.transform(df[[col]])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_arr = ohe.fit_transform(df[[col]])
            encoders[col] = ohe

        onehot_df = pd.DataFrame(
            onehot_arr,
            columns=[f"{col}_{cat}" for cat in encoders[col].categories_[0]],
            index=df.index
        )
        df = pd.concat([df, onehot_df], axis=1)

    # -------------------------
    # Pandas categorical type
    # -------------------------
    elif encoding == "categorical":
        df[col] = df[col].astype("category")
        if col not in encoders:
            encoders[col] = df[col].cat.categories.tolist()

    return df, encoders


def encode_application_records(df, encoding_type="label", encoders=None):
    
    categorical_cols = [
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "name_housing_type",
        "occupation_type"
    ]

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        df, encoders = encode_categories(
            df,
            col,
            encoding=encoding_type,
            encoders=encoders
        )
    if encoding_type in ["label", "onehot"]:
        df.drop(columns=categorical_cols, inplace=True)

    return df, encoders


# def drop_id_dupes(df):
#   df_sorted=df.sort_values('id')

#   def keep_row(id):
#     notna = id[id['occupation_type'].notna()]
#     if not notna.empty:
#       return notna.iloc[[0]] #keep first dup
#     else:
#       return id.iloc[[0]] #if everything is NaN, just keep first
#   df_dropped=df_sorted.groupby('id', group_keys=False).apply(keep_row)
#   return df_dropped.reset_index(drop=True)

def clean_application_records(raw, encoding_type="label", encoders=None):

    df = raw.copy()
    df.columns = df.columns.str.lower()

    # Handle duplicates
    df.drop_duplicates(['id'], keep='last')

    # ordinal encoding for cnt childen, age_binned and family size
    # Children
    df["cnt_children_encoded"] = df["cnt_children"].apply(lambda x: x if x in [0,1,2,3] else 4).astype(int)
    # Family size
    df["cnt_fam_members"] = df["cnt_fam_members"].astype(int)
    # df["cnt_fam_members_encoded"] = df["cnt_fam_members"].apply(lambda x: x if x in [1,2,3,4,5] else 6).astype(int)
    # df["cnt_fam_members_encoded"] = df["cnt_fam_members_encoded"] - df["cnt_fam_members_encoded"].min()

    # Age
    df["age"] = (-df["days_birth"] / 365).round(0).astype(int)
    age_bins = list(range(0,91,5)) + [float('inf')]
    age_labels = [f"{i}-{i+4}" for i in range(0,90,5)] + ["90+"]
    # df['age_binned'] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    # df['age_binned'] = pd.Categorical(df['age_binned'], categories=age_labels, ordered=True)
    # df['age_binned'] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    # df['age_binned_encoded'] = df['age_binned'].cat.codes
    # df['age_binned_encoded'] = df['age_binned'].cat.codes - df['age_binned'].cat.codes.min()

    # Employment
    df["days_employed"] = np.where(df["days_employed"] >= 0, -1, -df["days_employed"])
    df["months_employed"] = (df["days_employed"]/30.44).round(0).astype(int)
    df["months_employed"] = np.where(df["months_employed"] >= 0, df["months_employed"], -1)
    df["years_employed"] = (df["days_employed"]/365).round(0).astype(int)
    df["years_employed"] = np.where(df["years_employed"] >= 0, df["years_employed"], -1)
    df["employment_status_encoded"] = np.where(df["days_employed"]<0, 1, 0)


    # Binary flags
    df["flag_gender"] = df["code_gender"].map({'M':0, 'F':1})
    df["flag_own_realty"] = df["flag_own_realty"].map({'Y':1, 'N':0})
    df["flag_own_car"] = df["flag_own_car"].map({'Y':1, 'N':0})


    # Income
    df["amt_income_total_log"] = np.log1p(df["amt_income_total"])

    # Categorical
    df["occupation_type"] = df["occupation_type"].fillna("Unemployed")

    categorical_cols = [
        "name_income_type",
        "name_education_type",
        "name_family_status",
        "name_housing_type",
        "occupation_type"
    ]

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        df, encoders = encode_categories(
            df,
            col,
            encoding=encoding_type,
            encoders=encoders
        )
    

    drop_col = [
      "name_income_type"
      , "name_education_type"
      , "name_family_status"
      , "name_housing_type"
      , "occupation_type"
      , "amt_income_total"
      , "code_gender"
      , "days_birth"
      , "days_employed"]
    df.drop(columns=drop_col, inplace=True)

    return df, encoders
    

# clean credit records
def weighted_default_prop_decay(group, decay=0.1):
    weights = np.array([np.exp(-decay * i) for i in range(len(group)-1, -1, -1)])
    weighted_defaults = (group['default_flag'] * weights).sum()
    total_weights = weights.sum()
    return weighted_defaults / total_weights if total_weights != 0 else 0


def clean_credit_records(df):

    df = df.copy()
    df.columns = df.columns.str.lower()

    orig_map = df.groupby('id')['months_balance'].min()
    df['origination_month'] = df['id'].map(orig_map)

    df['status_num'] = pd.to_numeric(df['status'], errors='coerce')
    df['default_flag'] = df['status_num'].isin([2, 3, 4, 5]).astype(int)

    # Compute customer-level aggregates
    final_df = df.groupby(['id', 'origination_month']).apply(lambda x: pd.Series({
        'tenure': x.shape[0],
        # no. defaults / total records per customer
        'default_prop': x['default_flag'].mean(),
        # no. defaults is the
        'weighted_default_prop': weighted_default_prop_decay(x, decay=0.1),
        'risk_score': x['default_flag'].sum() * np.log1p(x.shape[0])
    })).reset_index()

    final_df['weighted_default_prop_tenure_norm'] = final_df['weighted_default_prop'] / np.log1p(final_df['tenure'])

    customer_vintage = df.groupby('id')['origination_month'].min().reset_index().abs()
    customer_vintage.columns = ['id', 'vintage']

    final_df = final_df.merge(customer_vintage, on='id', how='left')

    final_df['weighted_default_prop_vintage_norm'] = final_df.groupby('vintage')['weighted_default_prop'].transform(
      lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    return final_df


# In[ ]:


#splitting dataset
def split_credit_dataset(credit_records_raw):
  credit_records_copy = credit_records_raw.copy()
  credit_records_copy.columns = credit_records_copy.columns.str.lower()
  orig_map = credit_records_copy.groupby('id')['months_balance'].min()
  credit_records_copy['origination_month'] = credit_records_copy['id'].map(orig_map)

  # Keep only unique (id, origination_month) for CDF calculation
  unique_accounts = credit_records_copy.drop_duplicates(subset=['id', 'origination_month'])
  print(f"Total unique accounts: {len(unique_accounts)}. Starting to find cutoff point")
  # Determine cutoff month for ~80:20 split
  month_counts = unique_accounts['origination_month'].value_counts().sort_index(ascending=True)
  cumulative_counts = month_counts.cumsum()
  cdf = cumulative_counts / cumulative_counts.iloc[-1]
  cutoff_month = cdf.index[cdf >= 0.8].min()
  print(f"Cutoff month where CDF reaches 80%: {cutoff_month}")

  # Split into old/new unique accounts
  old_accounts = unique_accounts[unique_accounts['origination_month'] <= cutoff_month]
  new_accounts = unique_accounts[unique_accounts['origination_month'] > cutoff_month]

  old_count = len(old_accounts)
  new_count = len(new_accounts)
  total_count = len(unique_accounts)

  print(f"\n=== Split based on CDF 80% cutoff ===")
  print(f"Cutoff month: {cutoff_month} ({abs(cutoff_month)} months ago)")
  print(f"Old accounts (≤ month {cutoff_month}): {old_count:,} ({old_count/total_count*100:.1f}%)")
  print(f"New accounts (> month {cutoff_month}): {new_count:,} ({new_count/total_count*100:.1f}%)")
  print(f"Ratio (old/new): {old_count/new_count:.4f}")

  old_ids = set(old_accounts['id'])
  new_ids = set(new_accounts['id'])

  print('Splitting raw credit records')
  # Extract *all* raw credit records for those accounts
  old_accounts_credit_df = credit_records_copy[credit_records_copy['id'].isin(old_ids)]
  new_accounts_credit_df = credit_records_copy[credit_records_copy['id'].isin(new_ids)]

  return old_accounts_credit_df, new_accounts_credit_df, old_ids, new_ids

def split_application_dataset(application_records_raw, old_ids, new_ids):
  print('Splitting application dataset')
  application_records_df = application_records_raw.copy()
  application_records_df.columns = application_records_df.columns.str.lower()
  old_accounts_application = application_records_df[application_records_df['id'].isin(old_ids)]
  new_accounts_application = application_records_df[application_records_df['id'].isin(new_ids)]

  return old_accounts_application, new_accounts_application


# In[ ]:


# target engineering
# def create_target(df, weights=None, scaling_method = 'quantile'):
#     df = df.copy()

#     # Metrics to include
#     metrics = ['weighted_default_prop', 'default_prop', 'weighted_default_prop_tenure_norm', 'weighted_default_prop_vintage_norm']
#     # Equal weighting to each feature
#     if weights is None:
#         weights = {metric: 1/len(metrics) for metric in metrics}

#     # A) ROBUST (does not work well)
#     if scaling_method == 'robust':
#       scaler = RobustScaler()
#       for metric in metrics:
#           df[f'{metric}_scaled'] = scaler.fit_transform(df[[metric]])

#     # B) MANUAL QUANTILE CALC
#     elif scaling_method == 'manual_quantile':
#       for metric in metrics:
#         df[f'{metric}_scaled'] = df[metric].rank(pct=True)

#     # C) QUANTILE TRANSFORMER
#     else:
#       scaler = QuantileTransformer(output_distribution='uniform')
#       for metric in metrics:
#           df[f'{metric}_scaled'] = scaler.fit_transform(df[[metric]])



#     df['composite_risk_score'] = sum(df[f'{metric}_scaled'] * weight
#                                      for metric, weight in weights.items())

#     # Define threshold (80th percentile)
#     risk_threshold = df['composite_risk_score'].quantile(0.80)
#     df['multi_dim_target'] = (df['composite_risk_score'] > risk_threshold).astype(int)
#     df['customer_label'] = df['multi_dim_target'].map({0: 'Good Customer', 1: 'Bad Customer'})

#     return df, risk_threshold

def create_target(df, weights=None, scaling_method = 'quantile', risk_threshold = None):
  """
  Creates composite risk score & multi_dim_target from credit aggregates
  Finds 80% threshold from train set, and applies it to test set
  """
  df = df.copy()

  if risk_threshold is None:
    risk_threshold= df['default_prop'].quantile(0.8)

  df['multi_dim_target'] = (df['default_prop'] > risk_threshold).astype(int)


  df['customer_label'] = df['multi_dim_target'].map({0: 'Good Customer', 1: 'Bad Customer'})

  return df, risk_threshold



# Merge cleaned application and credit datasets
def merge_data(old_credit_df, new_credit_df, old_accounts_application_df, new_accounts_application_df):
  print(f'Engineering target variable to label data')
  old_simple, old_threshold_simple = create_target(old_credit_df, scaling_method = 'manual_quanitle')
  print(f'Completed old accounts labelling')
  new_simple, new_threshold_simple = create_target(new_credit_df, scaling_method = 'manual_quantile')
  print(f'Completed new accounts labelling')
  keep_cols = ['id', 'risk_score', 'multi_dim_target']
  old_accounts_credit_df = old_simple[keep_cols]
  new_accounts_credit_df = new_simple[keep_cols]
  print(f'Old accounts: {old_accounts_credit_df.shape}')
  print(f'New accounts: {new_accounts_credit_df.shape}')
  print(f'Old threshold: {old_threshold_simple}')
  print(f'New threshold: {new_threshold_simple}')
  print('Merging cleaned application and credit records')
  train = pd.merge(old_accounts_application_df, old_accounts_credit_df, on='id', how='inner').rename(columns = {'multi_dim_target':'label'})
  test = pd.merge(new_accounts_application_df, new_accounts_credit_df, on='id', how='inner').rename(columns = {'multi_dim_target':'label'})
  print(f'Train shape: {train.shape}')
  print(f'Test shape: {test.shape}')
  return train, test


# split into x and y
def X_y_split(train, test, target_col='label'):
#   drop_cols = ["days_birth", "amt_income_total", "days_employed", "years_employed", "cnt_fam_members", "flag_mobil"]
#   train_df = train.drop(columns=drop_cols)
#   test_df = test.drop(columns=drop_cols)

  train_df = train.copy()
  test_df  = test.copy()

  # train test split
  X_train_full = train_df.drop(columns=[target_col])
  y_train_full = train_df[target_col]
  X_test = test_df.drop(columns=[target_col])
  y_test = test_df[target_col]
  print('Completed X, y split')
  return X_train_full, y_train_full, X_test, y_test


def scaling_std(X_train, X_test, numeric_columns, ScalerType):
  X_train_scaled = X_train.copy()
  X_test_scaled = X_test.copy()

  # Drop 'id' before scaling
  if 'id' in X_train_scaled.columns:
      X_train_scaled = X_train_scaled.drop('id', axis=1)
  if 'id' in X_test_scaled.columns:
      X_test_scaled = X_test_scaled.drop('id', axis=1)

  scaler = ScalerType()

  # Use .loc with column names
  X_train_scaled[numeric_columns] = scaler.fit_transform(
      X_train_scaled[numeric_columns]
  )
  X_test_scaled[numeric_columns] = scaler.transform(
      X_test_scaled[numeric_columns]
  )

  return X_train_scaled, X_test_scaled



def data_pipeline(encode_type):
  print('Loading data')
  application_records_raw, credit_records_raw = load_data()
  print('Splitting data')
  old_accounts_credit_df, new_accounts_credit_df, old_ids, new_ids = split_credit_dataset(credit_records_raw)
  # Cleaning credit records to get respective columns required for target engineering
  print(f'Cleaning old accounts credit records - [Length: {old_accounts_credit_df.shape[0]}]')
  old_accounts_credit = clean_credit_records(old_accounts_credit_df)
  print(f'Cleaning new accounts credit records - [Length: {new_accounts_credit_df.shape[0]}]')
  new_accounts_credit = clean_credit_records(new_accounts_credit_df)
  print('Cleaning credit data completed')


  old_accounts_application_df, new_accounts_application_df = split_application_dataset(application_records_raw, old_ids, new_ids)
  print(f'Cleaning old accounts application records - [Length: {old_accounts_application_df.shape}]')
  old_accounts_application_df, old_encoders =  clean_application_records(old_accounts_application_df, encoding_type = encode_type)
  print(f'Cleaning new accounts appplication records, - [Length: {new_accounts_application_df.shape}]')
  new_accounts_application_df, new_encoders = clean_application_records(new_accounts_application_df, encoding_type = encode_type)
#   print('Encoding')
#   old_accounts_application_df, encoders = encode_application_records(old_accounts_application_df, encoding_type=encode_type)
#   new_accounts_application_df, _ = encode_application_records(new_accounts_application_df, encoding_type=encode_type, encoders=encoders)
#   print(f"Encoders: {encoders}")
#   print('Encoding type:', encode_type)

  print('Merging data')
  merged_train, merged_test = merge_data(old_accounts_credit, new_accounts_credit, old_accounts_application_df, new_accounts_application_df)
  print(merged_train.info())
  print(merged_test.info())

  X_train, y_train, X_test, y_test = X_y_split(merged_train, merged_test, target_col='label')

  numeric_columns = ["amt_income_total_log"
      , "age"
      , "months_employed"
      , "years_employed"
      , "cnt_fam_members"
      , "cnt_children"]


  X_train_std, X_test_std = scaling_std(
      X_train, X_test, numeric_columns, StandardScaler)


  print('Final train and test processing completed generated successfully')
  return X_train_std, y_train, X_test_std, y_test


