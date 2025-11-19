#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub
from kagglehub import KaggleDatasetAdapter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


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


def encode_categories(df, col, unknown_label="Unknown", encoders=None):
  """
  Apply LabelEncoder on categorical columns.
  "Unknown" encoded as -1
  """
  df[col] = df[col].astype(str)
  known_mask = df[col]!=unknown_label

  le=LabelEncoder()
  le.fit(df.loc[known_mask,col])

  # print encoding mappings

  print(f"\n encoding for column : {col}")
  for i, label in enumerate(le.classes_):
    print(f" {label} -> {i}")

  encoded=pd.Series(-1, index=df.index)
  encoded[known_mask]=le.transform(df.loc[known_mask,col])

  df[col+"_encoded"]=encoded

  # store encoded values for decoding alter on
  if encoders is not None:
        encoders[col] = le

  return df, encoders


# In[ ]:


def clean_app_rec(raw):
  """
  - lowercase column names
  - keep last entry of duplicated id
  - feature engineer age from days birth
  - feature engineer employment status & months & years employed from days employed
  - convert family size and count children to integers
  - binary flags
  - log transform income due to steep skewness
  - drop irrelevant columns
  """
  # column names lowercase
  raw.columns=raw.columns.str.lower()
  df=raw.copy()
  df.columns = df.columns.str.lower()

  # keep last duplicated id
  df.drop_duplicates(['id'], keep='last')

  ############ CLEANING

  ##### numerical
  # age (days_birth)
  df["age"] = (-df["days_birth"] / 365).round(0).astype(int)
  age_bins = list(range(0,91,5)) + [float('inf')]
  age_labels = [f"{i}-{i+4}" for i in range(0,90,5)] + ["90+"]
  df['age_binned'] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
  df['age_binned'] = pd.Categorical(df['age_binned'], categories=age_labels, ordered=True)
  df, encoders = encode_categories(df, "age_binned")

  # employment (days_employed)
  df["days_employed"] = np.where(df["days_employed"] >= 0, -1, -df["days_employed"])
  df["months_employed"] = (df["days_employed"]/30.44).round(0).astype(int)
  df["months_employed"] = np.where(df["months_employed"] >= 0, df["months_employed"], -1)
  df["years_employed"] = (df["days_employed"]/365).round(0).astype(int)
  df["years_employed"] = np.where(df["years_employed"] >= 0, df["years_employed"], -1)

  df["flag_employed"] = np.where(df["days_employed"]<0, 1, 0)

  # family size
  df["cnt_fam_members"] = df["cnt_fam_members"].astype(int)

  # children count
  df["cnt_children"] = df["cnt_children"].astype(int)

  # amt income
  df["amt_income_total_log"] = np.log1p(df["amt_income_total"])

  ##### binary
  df["flag_gender"] = df["code_gender"].map({'M':0, 'F':1})
  df["flag_own_realty"] = raw["flag_own_realty"].map({'Y':1, 'N':0})
  df["flag_own_car"] = raw["flag_own_car"].map({'Y':1, 'N':0})

  ##### categorical
  df["occupation_type"] = df["occupation_type"].fillna("Unemployed")

  categorical_col = ["name_income_type", "name_education_type", "name_family_status", "name_housing_type", "occupation_type"]
  for col in categorical_col:
    df, encoders = encode_categories(df, col)

  ########## drop columns
  drop_col = [
      "name_income_type"
      , "name_education_type"
      , "name_family_status"
      , "name_housing_type"
      , "occupation_type"
      , "age_binned"
      , "amt_income_total"
      , "code_gender"
      , "days_birth"
      , "days_employed"]
  df.drop(columns=drop_col, inplace=True)

  return df


# In[ ]:


def weighted_default_prop_decay(group, decay=0.1):
  """
  Exponential decay weighting for defualt_prop
  Newer months get larger weights
  """
  group = group.sort_values('months_balance', ascending=True)
  weights = np.exp(-decay * np.abs(group['months_balance']))
  weighted_avg = np.sum(group['default_flag'] * weights) / np.sum(weights)
  return weighted_avg


def clean_credit_records(df):
  """
  Aggregates credit record to customer level
  **EXCLUDE risk_score later on to prevent data leakage
  """
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


def split_credit_dataset(credit_records_raw):
  """
  80:20 split by origination_month CDF
  Returs:
    - old_accounts_credit_df
    - new_accounts_credit_df
    - old_ids
    - new_ids
  """
  credit_records_copy = credit_records_raw.copy()
  credit_records_copy.columns = credit_records_copy.columns.str.lower()
  orig_map = credit_records_copy.groupby('id')['months_balance'].min()
  credit_records_copy['origination_month'] = credit_records_copy['id'].map(orig_map)

  # Keep only unique (id, origination_month) for CDF calculation
  unique_accounts = credit_records_copy.drop_duplicates(subset=['id', 'origination_month'])

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

  # Extract *all* raw credit records for those accounts
  old_accounts_credit_df = credit_records_copy[credit_records_copy['id'].isin(old_ids)]
  new_accounts_credit_df = credit_records_copy[credit_records_copy['id'].isin(new_ids)]

  return old_accounts_credit_df, new_accounts_credit_df, old_ids, new_ids

def split_application_dataset(application_records_raw, old_ids, new_ids):
  application_records_df = clean_app_rec(application_records_raw)
  old_accounts_application = application_records_df[application_records_df['id'].isin(old_ids)]
  new_accounts_application = application_records_df[application_records_df['id'].isin(new_ids)]

  return old_accounts_application, new_accounts_application


# In[ ]:


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


# In[ ]:


status_map = {
    'X':-1,
    'C': -1,
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5
}

status_to_days = {
        -1: 0,
        0: 29,
        1: 59,
        2: 89,
        3: 119,
        4: 149,
        5: 150
    }
def feature_engineer(df):
    df = df.copy()
    df['status_num'] = df['status'].map(status_map)
    df = df.sort_values(['id', 'months_balance'])

    # max days past due
    max_status = df.groupby('id')['status_num'].max().reset_index()
    max_status['max_days_past_due'] = max_status['status_num'].map(status_to_days)

    # longest fault/non-default streaks
    df['is_default'] = (df['status_num'] >= 2).astype(int)

    # Streak groups: whenever is_default changes
    df['streak_group'] = (df['is_default'] != df.groupby('id')['is_default'].shift()).cumsum()

    streaks = (
        df.groupby(['id', 'streak_group'])['is_default']
          .agg(streak_length='size', is_default='first')
          .reset_index()
    )

    longest_streaks = (
        streaks.pivot_table(index='id', columns='is_default', values='streak_length', aggfunc='max', fill_value=0)
        .rename(columns={0: 'longest_non_default_streak', 1: 'longest_default_streak'})
        .reset_index()
    )

    # consecutive defaults ins last 6m
    def consecutive_defaults_last_6m_vec(status_series):
        recent = status_series.tail(6).map(status_map).to_numpy()
        defaults = (recent >= 2).astype(int)
        return np.cumprod(defaults[::-1]).sum()

    consec_default_6m = (
        df.groupby('id')['status']
          .agg(consecutive_defaults_6m=consecutive_defaults_last_6m_vec)
          .reset_index()
    )

    # recency of last bad status
    def recency_last_bad(months, status_series):
        status_num = status_series.map(status_map)
        bad_months = months[status_num >= 2]
        if bad_months.empty:
            return np.nan
        else:
            return abs(bad_months.max() - months.max())

    recency_bad = (
        df.groupby('id').apply(lambda x: recency_last_bad(x['months_balance'], x['status']))
        .reset_index(name='recency_last_bad_status')
    )

    # merge all features
    features = max_status[['id','max_days_past_due']].merge(
        longest_streaks, on='id', how='left'
    ).merge(
        consec_default_6m, on='id', how='left'
    ).merge(
        recency_bad, on='id', how='left'
    )

    # Fill in missing values
    features[['longest_non_default_streak','longest_default_streak','consecutive_defaults_6m']] = \
        features[['longest_non_default_streak','longest_default_streak','consecutive_defaults_6m']].fillna(0)

    features['recency_last_bad_status'] = features['recency_last_bad_status'].fillna(-1)

    return features

# engineered_cols = [
#     'longest_non_default_streak' # good streak
#     ,'longest_default_streak' # bad streak
#     ,'consecutive_defaults_6m' # recent bad behaviour
#     , 'max_days_past_due' # worst delinquency
#     , 'recency_last_bad_status'] # how long since the last default


# In[ ]:


def merge_data(old_credit_df
               , new_credit_df
               , old_accounts_credit_df
               , new_accounts_credit_df
               , old_accounts_application_df
               , new_accounts_application_df):
  old_eng = feature_engineer(old_accounts_credit_df)
  new_eng = feature_engineer(new_accounts_credit_df)

  old_credit_full = old_credit_df.merge(old_eng, on="id", how="left")
  new_credit_full = new_credit_df.merge(new_eng, on="id", how="left")


  old_simple, old_threshold_simple = create_target(old_credit_full, scaling_method = 'manual_quanitle')
  new_simple, new_threshold_simple = create_target(new_credit_full, scaling_method = 'manual_quantile')

  credit_cols = [
      'id'
      , 'multi_dim_target'
      , 'max_days_past_due'
      , 'longest_non_default_streak'
      , 'longest_default_streak'
      , 'consecutive_defaults_6m'
      , 'recency_last_bad_status']

  old_accounts_credit_df = old_simple[credit_cols]
  new_accounts_credit_df = new_simple[credit_cols]

  train = pd.merge(old_accounts_application_df, old_accounts_credit_df, on='id', how='inner').rename(columns = {'multi_dim_target':'label'})
  test = pd.merge(new_accounts_application_df, new_accounts_credit_df, on='id', how='inner').rename(columns = {'multi_dim_target':'label'})

  return train, test


# In[ ]:


def get_feat_cols(drop_col, categorical_col, binary_col, numerical_col):
  feature_cols = [c for c in (categorical_col + binary_col + numerical_col) if c not in drop_col]
  return feature_cols

def make_sampled_sets(merged_train, merged_test, feature_cols, label_col="label"):
  X_train = merged_train[feature_cols].copy()
  y_train = merged_train[label_col].copy()

  X_test = merged_test[feature_cols].copy()
  y_test = merged_test[label_col].copy()

  ros = RandomOverSampler(random_state=42)
  X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

  rus = RandomUnderSampler(random_state=42)
  X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

  smote = SMOTE(random_state=42, k_neighbors=5)
  X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

  return X_train, y_train, X_test, y_test, X_train_ros, y_train_ros, X_train_rus, y_train_rus, X_train_smote, y_train_smote


# In[ ]:


def data_pipeline():
  print('==============================1 Loading data')
  application_records_raw, credit_records_raw = load_data()

  print('==============================2 Splitting data based on old & new')
  old_credit_raw, new_credit_raw, old_ids, new_ids = split_credit_dataset(credit_records_raw)

  print('==============================3 Aggregating credit')
  old_credit_agg = clean_credit_records(old_credit_raw)
  new_credit_agg = clean_credit_records(new_credit_raw)

  print('==============================4 Defining target variable based on train')
  old_labeled, risk_threshold = create_target(old_credit_agg)
  new_labeled, _ = create_target(new_credit_agg, risk_threshold=risk_threshold)

  print('==============================5 Splitting application record based on old & new')
  old_app, new_app = split_application_dataset(application_records_raw, old_ids, new_ids)

  print('==============================6 Merging & feature engineering')
  merged_train, merged_test = merge_data(
      old_labeled,
      new_labeled,
      old_credit_raw,
      new_credit_raw,
      old_app,
      new_app
  )

  print('==============================7 Dropping columns')
  drop_col = [
    "flag_mobil"
    , "years_employed"
    , "cnt_children"
    ]
  categorical_col = [
      "name_income_type_encoded"
      , "name_education_type_encoded"
      , "name_family_status_encoded"
      , "name_housing_type_encoded"
      , "occupation_type_encoded"
      , "age_binned_encoded"]

  binary_col = [
      "flag_gender"
      , "flag_own_car"
      , "flag_own_realty"
      , "flag_mobil"
      , "flag_work_phone"
      , "flag_phone"
      , "flag_email"
      , "flag_employed"]

  numerical_col = [
      "amt_income_total_log"
      , "age"
      , "months_employed"
      , "years_employed"
      , "cnt_fam_members"
      , "cnt_children"
      , 'max_days_past_due'
      , 'longest_non_default_streak'
      , 'longest_default_streak'
      , 'consecutive_defaults_6m'
      , 'recency_last_bad_status']
  feature_cols = get_feat_cols(drop_col, categorical_col, binary_col, numerical_col)

  print('=========================8 SAMPLING')
  X_train, y_train, X_test, y_test, X_train_ros, y_train_ros, X_train_rus, y_train_rus, X_train_smote, y_train_smote = make_sampled_sets(merged_train, merged_test, feature_cols)

  print('=========================PIPELINE COMPLETED')
  result = {
      "feature_cols": feature_cols,
      "categorical_col": categorical_col,
      "binary_col": binary_col,
      "numerical_col": numerical_col,
      "train": {
          "X": X_train,
          "y": y_train
      },
      "test": {
          "X": X_test,
          "y": y_test
      },
      "resampled": {
          "ros": (X_train_ros,y_train_ros),
          "rus": (X_train_rus, y_train_rus),
          "smote": (X_train_smote, y_train_smote)
      }
  }

  return result


# In[23]:


if __name__ == '__main__':
  data=data_pipeline()

  X_train = data["train"]["X"]
  y_train = data["train"]["y"]
  X_test = data["test"]["X"]
  y_test = data["test"]["y"]

  X_train_ros, y_train_ros = data["resampled"]["ros"]
  X_train_rus, y_train_rus = data["resampled"]["rus"]
  X_train_smote, y_train_smote = data["resampled"]["smote"]

  feature_cols = data["feature_cols"]
  categorical_col = data["categorical_col"]
  binary_col = data["binary_col"]
  numerical_col = data["numerical_col"]

  print(feature_cols)
  print(categorical_col)
  print(binary_col)
  print(numerical_col)

  print(f"Train: X_train = {X_train.shape}, y_train = {y_train.shape}")
  print(f"Test : X_test  = {X_test.shape}, y_test  = {y_test.shape}")

  print("\n-- SANITY CHECKS ON RESAMPLED SETS --")
  print(f"ROS   : X = {X_train_ros.shape}, y = {y_train_ros.shape}")
  print(f"RUS   : X = {X_train_rus.shape}, y = {y_train_rus.shape}")
  print(f"SMOTE : X = {X_train_smote.shape}, y = {y_train_smote.shape}")

  # ---- ASSERTIONS ----
  assert len(X_train) == len(y_train), "❌ ERROR: X_train and y_train length mismatch!!!!!!!!"
  assert len(X_test) == len(y_test), "❌ ERROR: X_test and y_test length mismatch!!!!!!!!"
  assert len(X_train_ros) == len(y_train_ros), "❌ ERROR: ROS set mismatch!!!!!!!!"
  assert len(X_train_rus) == len(y_train_rus), "❌ ERROR: RUS set mismatch!!!!!!!!"
  assert len(X_train_smote) == len(y_train_smote), "❌ ERROR: SMOTE set mismatch!!!!!!!!"

  print("\nAll sanity checks passed ✓")

