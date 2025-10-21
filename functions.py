#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub
from kagglehub import KaggleDatasetAdapter
from  sklearn.preprocessing import LabelEncoder
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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


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


# In[5]:


# application record clean functions
def encode_categories(df, col, unknown_label="Unknown", encoders=None):
  df[col] = df[col].astype(str)
  known_mask = df[col]!=unknown_label

  le=LabelEncoder()
  le.fit(df.loc[known_mask,col])

  encoded=pd.Series(-1, index=df.index)
  encoded[known_mask]=le.transform(df.loc[known_mask,col])

  df[col+"_encoded"]=encoded

  # store encoded values for decoding alter on
  if encoders is not None:
        encoders[col] = le
  return df, encoders

def drop_id_dupes(df):
  df_sorted=df.sort_values('id')

  def keep_row(id):
    notna = id[id['occupation_type'].notna()]
    if not notna.empty:
      return notna.iloc[[0]] #keep first dup
    else:
      return id.iloc[[0]] #if everything is NaN, just keep first
  df_dropped=df_sorted.groupby('id', group_keys=False).apply(keep_row)
  return df_dropped.reset_index(drop=True)

def clean_application_records(raw):
    print('Starting to clean application records')
    raw.columns=raw.columns.str.lower()
    df=raw.copy()
    df.columns = df.columns.str.lower()

    dupes_df = df[df['id'].duplicated(keep=False)].sort_values(by='id')
    dupes_df = drop_id_dupes(dupes_df)
    non_dupes = df[~df['id'].duplicated(keep=False)]
    df = pd.concat([non_dupes, dupes_df], ignore_index=True)

    # Children
    df["cnt_children_encoded"] = df["cnt_children"].apply(lambda x: x if x in [0,1,2,3] else 4).astype(int)

    # Age
    df["age"] = (-df["days_birth"] / 365).round(0).astype(int)
    age_bins = list(range(0,91,5)) + [float('inf')]
    age_labels = [f"{i}-{i+4}" for i in range(0,90,5)] + ["90+"]
    df['age_binned'] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    df['age_binned'] = pd.Categorical(df['age_binned'], categories=age_labels, ordered=True)
    df, encoders = encode_categories(df, "age_binned")

    # Employment
    df["days_employed"] = np.where(df["days_employed"] >= 0, -1, -df["days_employed"])
    df["months_employed"] = (df["days_employed"]/30.44).round(0).astype(int)
    df["months_employed"] = np.where(df["months_employed"] >= 0, df["months_employed"], -1)
    df["years_employed"] = (df["days_employed"]/365).round(0).astype(int)
    df["years_employed"] = np.where(df["years_employed"] >= 0, df["years_employed"], -1)

    df["employment_status_encoded"] = np.where(df["days_employed"]<0, 1, 0)

    # Family size
    df["cnt_fam_members"] = df["cnt_fam_members"].astype(int)
    df["cnt_fam_members_encoded"] = df["cnt_fam_members"].apply(lambda x: x if x in [1,2,3,4,5] else 6).astype(int)

    # Encode categorical variables
    cat_cols = ["name_income_type", "name_education_type", "name_family_status", "name_housing_type"]
    for col in cat_cols:
        df, encoders = encode_categories(df, col)

    # Occupation type
    df["occupation_type"] = df["occupation_type"].fillna("Unemployed")
    df, encoders = encode_categories(df, "occupation_type")

    # Binary flags
    df["gender_encoded"] = df["code_gender"].map({'M':0, 'F':1})
    df["flag_own_realty_encoded"] = df["flag_own_realty"].map({'Y':1, 'N':0})
    df["flag_own_car_encoded"] = df["flag_own_car"].map({'Y':1, 'N':0})

    # Income log transform
    df["amt_income_total_log"] = np.log1p(df["amt_income_total"])

    # Employment consistency check
    df.loc[df["employment_status_encoded"]==0, "occupation_type_encoded"] = -1
    df.loc[df["employment_status_encoded"]==1, "occupation_type_encoded"] = df.loc[df["employment_status_encoded"]==1, "occupation_type_encoded"].replace(-1,0)

    # Drop original object columns
    print('Dropping object columns')
    object_cols = df.select_dtypes(include='object').columns
    df.drop(columns=object_cols, inplace=True)
    # print(object_cols)
    print('Completed cleaning raw application records')
    # Scale numeric columns except binaries and label-encoded categories if needed (to be done later)
    return df


# In[6]:


# clean credit records
def weighted_default_prop_decay(group, decay=0.1):

    group = group.sort_values('months_balance', ascending=True)
    weights = np.exp(-decay * np.abs(group['months_balance']))
    weighted_avg = np.sum(group['default_flag'] * weights) / np.sum(weights)
    return weighted_avg


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


# In[7]:


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
  application_records_df = clean_application_records(application_records_raw)
  old_accounts_application = application_records_df[application_records_df['id'].isin(old_ids)]
  new_accounts_application = application_records_df[application_records_df['id'].isin(new_ids)]

  return old_accounts_application, new_accounts_application


# In[8]:


# target engineering
def create_target(df, weights=None, scaling_method = 'quantile'):
    df = df.copy()

    # Metrics to include
    metrics = ['weighted_default_prop', 'default_prop', 'weighted_default_prop_tenure_norm', 'weighted_default_prop_vintage_norm']
    # Equal weighting to each feature
    if weights is None:
        weights = {metric: 1/len(metrics) for metric in metrics}

    # A) ROBUST (does not work well)
    if scaling_method == 'robust':
      scaler = RobustScaler()
      for metric in metrics:
          df[f'{metric}_scaled'] = scaler.fit_transform(df[[metric]])

    # B) MANUAL QUANTILE CALC
    elif scaling_method == 'manual_quantile':
      for metric in metrics:
        df[f'{metric}_scaled'] = df[metric].rank(pct=True)

    # C) QUANTILE TRANSFORMER
    else:
      scaler = QuantileTransformer(output_distribution='uniform')
      for metric in metrics:
          df[f'{metric}_scaled'] = scaler.fit_transform(df[[metric]])



    df['composite_risk_score'] = sum(df[f'{metric}_scaled'] * weight
                                     for metric, weight in weights.items())

    # Define threshold (80th percentile)
    risk_threshold = df['composite_risk_score'].quantile(0.80)
    df['multi_dim_target'] = (df['composite_risk_score'] > risk_threshold).astype(int)
    df['customer_label'] = df['multi_dim_target'].map({0: 'Good Customer', 1: 'Bad Customer'})

    return df, risk_threshold


# In[14]:


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



# In[15]:


# split into x and y
def X_y_split(train, test, target_col='label'):
  drop_cols = ["days_birth", "amt_income_total", "days_employed", "years_employed", "cnt_fam_members", "flag_mobil"]
  train_df = train.drop(columns=drop_cols)
  test_df = test.drop(columns=drop_cols)

  train = train_df.copy()
  test = test_df.copy()

  # train test split
  X_train_full = train.drop(columns=[target_col])
  y_train_full = train[target_col]
  X_test = test.drop(columns=[target_col])
  y_test = test[target_col]
  print('Completed X, y split')
  return X_train_full, y_train_full, X_test, y_test


# In[13]:


def data_pipeline():
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

  print('Merging data')
  merged_train, merged_test = merge_data(old_accounts_credit, new_accounts_credit, old_accounts_application_df, new_accounts_application_df)
  print(merged_train.info())
  print(merged_test.info())

  print('Final train and test processing completed generated successfully')
  return merged_train, merged_test

