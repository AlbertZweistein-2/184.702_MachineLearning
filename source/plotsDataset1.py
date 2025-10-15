import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
folder_path = '../datasets/UJIndoorLoc/'
UJI_training_file = folder_path + 'trainingData.csv'
UJI_validation_file = folder_path + 'validationData.csv'
UJI_training = pd.read_csv(UJI_training_file)
UJI_validation = pd.read_csv(UJI_validation_file)


print(f"There are {UJI_training.shape[0]} instances in the training set and {UJI_validation.shape[0]} in the validation/testing set.")

#Nr of Wifi Access Points
WAP_columns = [col for col in UJI_training.columns if col.startswith('WAP')]
print(f"There are {UJI_training.shape[1]} features in the dataset. The first {len(WAP_columns)} are the Wifi Access Points (WAPs), labeled WAP001 to WAP520. The remaining features are: Longitude, Latitude, Floor, BuildingID, SpaceID, RelativePosition, UserID, PhoneID, and Timestamp.")
print(f"There are {UJI_training.isnull().sum().sum()} missing values in the training set and {UJI_validation.isnull().sum().sum()} in the validation/testing set.")
print(f"There are {UJI_training.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])['SPACEID'].nunique().sum()} unique spaces in the training set, which is more than the 123 unique spaceIDs, because some spaceIDs are present in multiple buildings and multiple floors!.")


building_counts = UJI_training['BUILDINGID'].unique()
floors_per_building = UJI_training.groupby('BUILDINGID')['FLOOR'].unique()

floor_counts = UJI_training.groupby(['BUILDINGID', 'FLOOR']).size()

floor_counts_dict = {}
for building in floor_counts.index.levels[0]:
    floor_counts_dict[building] = {}
    for floor in floor_counts[building].index:
        count = floor_counts[building][floor]
        floor_counts_dict[building][floor] = int(count)  

records = [
    {"BUILDINGID": b, "FLOOR": f, "count": c}
    for b, floors in floor_counts_dict.items()
    for f, c in floors.items()
]
floor_df = pd.DataFrame(records)


floor_df['BUILDINGID'] = floor_df['BUILDINGID'].astype(int)
floor_df['FLOOR'] = floor_df['FLOOR'].astype(int)

plt.figure(figsize=(10, 4))
ax = sns.barplot(data=floor_df, x='BUILDINGID', y='count', hue='FLOOR', palette='tab10')
ax.set_title('Number of Samples per Floor and Building')
ax.set_xlabel('Building ID')
ax.set_ylabel('Sample Count')

for container in ax.containers:
    ax.bar_label(container, padding=2, fontsize=8)

ax.legend(title='Floor', frameon=True)
plt.tight_layout()
plt.savefig('sample_distribution_per_floor_building.png', dpi=300)


UJI_training_positions_df = UJI_training[['BUILDINGID', 'LONGITUDE', 'LATITUDE', 'FLOOR', 'SPACEID']]
spaces_df = UJI_training_positions_df.drop_duplicates(subset=['BUILDINGID', 'FLOOR', 'SPACEID'])
# spaces_df = UJI_training_positions_df
spaces_df = spaces_df.sort_values(by=['BUILDINGID', 'FLOOR', 'SPACEID'])
min_longitude = spaces_df['LONGITUDE'].min()
max_longitude = spaces_df['LONGITUDE'].max()
min_latitude = spaces_df['LATITUDE'].min()
max_latitude = spaces_df['LATITUDE'].max()
spaces_df['MAX_LONGITUDE'] = UJI_training.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])['LONGITUDE'].max().values
spaces_df['MIN_LONGITUDE'] = UJI_training.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])['LONGITUDE'].min().values
spaces_df['MAX_LATITUDE'] = UJI_training.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])['LATITUDE'].max().values
spaces_df['MIN_LATITUDE'] = UJI_training.groupby(['BUILDINGID', 'FLOOR', 'SPACEID'])['LATITUDE'].min().values
# spaces_df.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)
color_mapping = {0: 'red', 1: 'blue', 2: 'green'}

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
# Plot 1: Scatter x: Longitude, y: Floor, color: BuildingID
sns.scatterplot(
    data=UJI_training,
    x='LONGITUDE',
    y='FLOOR',
    hue='BUILDINGID',
    ax=axs[0],
    legend=False
)
axs[0].set_title('Front View of the unique spaces in the buildings')
axs[0].set(xlabel=None, ylabel='Floor', yticks=range(0, 5), xticks=[])
# Plot 2: Scatter x: Longitude, y: Latitude, color: BuildingID, shape: Floor
sns.scatterplot(
    data=UJI_training,
    x='LONGITUDE',
    y='LATITUDE',
    hue='BUILDINGID',
    ax=axs[1]
)
axs[1].set_title('Top View of the unique spaces in the buildings')
axs[1].set(xlabel='Longitude', ylabel='Latitude', yticks=np.arange(min_latitude, max_latitude, step=50))
fig.savefig('building_floor_space_distribution.png', dpi=300)
