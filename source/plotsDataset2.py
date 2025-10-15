import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

folder_path = '../datasets/MI/'

mi_features_file = folder_path + 'mi_features.csv'
mi_targets_file = folder_path + 'mi_targets.csv'
mi_variables_file = folder_path + 'mi_variables.csv'
mi_metadata_file = folder_path + 'mi_metadata.json'

mi_features = pd.read_csv(mi_features_file, index_col=0)
mi_targets = pd.read_csv(mi_targets_file, index_col=0)
mi_variables = pd.read_csv(mi_variables_file, index_col=0)

with open(mi_metadata_file, 'r') as f:
    mi_metadata = json.load(f)

print(f"""
Dataset Name:          {mi_metadata["name"]}
Abstract:              {mi_metadata["abstract"]}
Area:                  {mi_metadata["area"]}
Task:                  {mi_metadata["tasks"][0]}
Last updated:          {mi_metadata["last_updated"]}
Has missing values:    {mi_metadata["has_missing_values"]}
Number of Instances:   {mi_metadata["num_instances"]}
Number of Features:    {mi_metadata["num_features"]}
Number of Targets:     {len(mi_metadata["target_col"])}
""")


# TARGET VARIABLES
nr_binary_targets = sum(mi_variables[mi_variables['role'].str.lower() == 'target']['type'].str.lower() == 'binary')
nr_categorical_targets = sum(mi_variables[mi_variables['role'].str.lower() == 'target']['type'].str.lower() == 'categorical')

print(f"There are {nr_binary_targets} binary target variables and {nr_categorical_targets} categorical target variables.")

mi_target_variables = mi_variables[mi_variables['role'].str.lower() == 'target'].copy()
mi_target_variables.set_index('name', inplace=True)
mi_target_variables["count"] = 0
for i in mi_target_variables.index:
    if mi_target_variables.loc[i, "type"].lower() == 'binary':
        mi_target_variables.loc[i, "count"] = mi_targets[i].sum()
    else:
        mi_target_variables.loc[i, "count"] = mi_targets[i].count()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 4))
data = mi_target_variables[mi_target_variables["type"].str.lower() == "binary"].sort_values(by="count", ascending=False)
ax = sns.barplot(x="count", y="description", data=data, orient="h")
ax.set_title("Occurence of binary target variables to be True")
ax.set_xlabel("Count")
ax.set_ylabel("Target Variables")
abs_values = data["count"].values
ax.bar_label(container=ax.containers[0], labels=abs_values)
plt.savefig('mi_binary_target_variables.png', dpi=300, bbox_inches='tight')

LET_IS_MAPPING = {
    0: "unknonw (alive)",
    1: "cardiogenic shock",
    2: "pulmonary edema",
    3: "myocardial rupture",
    4: "progress of congestive heart failure",
    5: "thromboembolism",
    6: "asystole",
    7: "ventricular fibrillation"
}
plt.figure(figsize=(10, 4))
ax = sns.countplot(data=mi_targets, y='LET_IS', order=mi_targets['LET_IS'].value_counts().index)
ax.set_yticks(ticks=list(LET_IS_MAPPING.keys()), labels=list(LET_IS_MAPPING.values()), ha='right')
ax.set_title('Distribution of the categorical target variable lethal outcome cause (LET_IS)')
ax.set_ylabel('Lethal outcome cause')
ax.set_xlabel('Count')
ax.set_xlim(0, 1600)
abs_values = mi_targets['LET_IS'].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=abs_values)
plt.savefig('mi_lethal_outcome_cause.png', dpi=300, bbox_inches='tight')


# FEATURE VARIABLES
mi_feature_variables = mi_variables[mi_variables['role'].str.lower() == 'feature']
mi_feature_variables.set_index('name', inplace=True)

print(f"There are {mi_feature_variables['missing_values'].value_counts()['yes']} feature variables with missing values and {mi_feature_variables['missing_values'].value_counts()['no']} without missing values.")

plt.figure(figsize=(10, 4))
ax = sns.countplot(data=mi_feature_variables, y='type', order=mi_feature_variables['type'].value_counts().index)
ax.set_xlabel('Count')
ax.set_ylabel('Feature Type')
ax.set_title('Distribution of Types of Feature Variables')
abs_values = mi_feature_variables['type'].value_counts().values
ax.bar_label(container=ax.containers[0], labels=abs_values)
plt.savefig('mi_feature_variable_types.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 4))
plt.hist(mi_features[mi_features["SEX"] == 1]["AGE"], label='Male', alpha=0.8, bins=20)
plt.hist(mi_features[mi_features["SEX"] == 0]["AGE"], label='Female', alpha=0.6, bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Sex')
plt.legend()
plt.savefig('mi_age_distribution_by_sex.png', dpi=300, bbox_inches='tight')


