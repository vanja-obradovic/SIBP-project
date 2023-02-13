from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pylab as pl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime


class data_cleanUp():
    def __init__(self):
        self.current_year = None
        self.cat_columns = None
        self.average_sqm_price_district = None
        self.neto2bruto = None
        self.kitchen_lifeSq_ratio = None

    def fit(self, data):
        self.current_year = datetime.datetime.now().year
        self.cat_columns = data.select_dtypes('object').columns
        self.neto2bruto = data.LifeSquare/data.Square
        self.kitchen_lifeSq_ratio = data.KitchenSquare/data.LifeSquare

        # Area normalization
        data.loc[(data.Square < 1.3), 'Square'] = data['Square']*100
        data.loc[(data.Square <= 13), 'Square'] = data['Square']*10
        data.loc[(data.Square > 300), 'Square'] = data['Square']/10

        # Room count normalization
        data.loc[data.Rooms == 0, 'Rooms'] = 1
        data.loc[(data.Rooms > 6) & (data.LifeSquare <
                 150), 'Rooms'] = data.Rooms.mean()

        # House floor normalization 
        data.loc[(data.Floor > data.HouseFloor),
                 'HouseFloor'] =  data.loc[(data.Floor > data.HouseFloor)].Floor

        #Convert HouseYear to HouseAge
        data.loc[(data.HouseYear > self.current_year),
                 'HouseYear'] = self.current_year-1
        data.insert(data.columns.get_loc('HouseYear'), 'HouseAge',
                    self.current_year - data['HouseYear'])
        data.drop('HouseYear', axis=1, inplace=True)

        # Convert text to numerical
        data = pd.get_dummies(data, columns=self.cat_columns)

        # Fill Healthcare_1 blanks
        data = data.sort_values(by='DistrictId')
        data.Healthcare_1.fillna(method='pad', inplace=True)

        # Fill LifeSquare blanks
        data = data.sort_values(by='Square')
        data.LifeSquare.fillna(method='pad', inplace=True)

        # Normalize LifeSquare
        data.loc[(data.LifeSquare/data.Square < self.neto2bruto.quantile(0.25)) |
                 (data.LifeSquare/data.Square > 0.85), 'LifeSquare'] = data.loc[(data.LifeSquare/data.Square < self.neto2bruto.quantile(0.25)) |
                 (data.LifeSquare/data.Square > 0.85)].Square*self.neto2bruto.mean()

        # Normalize KitchenSquare
        data.loc[(data.KitchenSquare/data.LifeSquare < 0.05) |
                 (data.KitchenSquare/data.LifeSquare > 0.4), 'KitchenSquare'] = data.loc[(data.KitchenSquare/data.LifeSquare < 0.05) |
                 (data.KitchenSquare/data.LifeSquare > 0.4)].LifeSquare*self.kitchen_lifeSq_ratio.mean()
        # data.sort_index(inplace=True)

        data = data.sort_values(by='DistrictId')
        data.insert(1, 'SqMeterPrice', data['Price']/data['Square'])
        self.average_sqm_price_district = data.groupby(by=['DistrictId'])[
            'SqMeterPrice'].mean().rename('AvgDistSqmPrice', inplace=True)
        data.drop('SqMeterPrice', axis=1, inplace=True)
        data.insert(1, 'AvgDistSqmPrice', self.average_sqm_price_district)
        data = data.sort_values(by='DistrictId')
        data['AvgDistSqmPrice'] = data.groupby(
            'DistrictId')['AvgDistSqmPrice'].bfill().ffill()

        data['DistrictId'] = data['DistrictId'].astype('object')
        data['Id'] = data['Id'].astype('object')
        data['Price'] = data.pop('Price')

        return data


TRAIN_PATH = "data/train.csv"

data = pd.read_csv(TRAIN_PATH)


print("\n------------------------------ PLOTTING AND DATA ANALYSIS ------------------------------")

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)

# General infromation
print("First 5 rows:", end='\n\n')
print(data.head(), end='\n\n')
print("Last 5 rows:", end='\n\n')
print(data.tail(), end='\n\n')
print("Info about the dataset:", end='\n\n')
print(data.info(), end='\n\n')
print("General statistic information about attributes:", end='\n\n')
pd.set_option('display.max_columns', 10)
print(data.describe(), end='\n\n')
print("Number of missing values for each column:", end='\n\n')
print(data.isna().sum(), end='\n\n')

# Attribute histograms
data_num_features = data.select_dtypes(include=['float32', 'float64', 'int8', 'int16', 'int32', 'int64'])
data_num_features.drop(['Price', 'Id', 'DistrictId'], axis=1, inplace=True)
data_num_features.hist(figsize=(16,16), bins=45, color="#14a1d9")
pl.suptitle("Overview of numerical input attributes before normalization", fontsize=18, fontweight="bold")
pl.tight_layout(pad=3.5)
plt.savefig('Attribute overview before normalization.png', dpi=250)
plt.show()

data.insert(1, "SqmPrice", data.Price/data.Square)

# Attributes with largest deviations
plt.figure(figsize=(8, 12))
sns.boxplot(data=data[['Square', 'LifeSquare', 'KitchenSquare', 'Rooms']], orient='h')
plt.xscale('symlog')
plt.xlim(left=-0.5)
plt.title('Overview of attributes with largest deviation')
plt.savefig('Attributes with largest deviation.png', dpi=250)
# Sqm price before normalization
plt.figure(figsize=(8, 12))
sns.boxplot(data=data.SqmPrice, orient='h')
plt.title('Price per square meter before data engineering')
plt.savefig('Sqm price before data engineering.png', dpi=250)
plt.show()

data.drop('SqmPrice', axis=1, inplace=True)

print("\n------------------------------ DATA ENGINEERING ------------------------------")

# Data normalization
cleaner = data_cleanUp()
clean = cleaner.fit(data)

# Attribute histograms after normalization
data_num_features = clean.select_dtypes(include=['float32', 'float64', 'int8', 'int16', 'int32', 'int64'])
data_num_features.drop(['Price'], axis=1, inplace=True)
data_num_features.hist(figsize=(16,16), bins=45, color="#14a1d9")
pl.suptitle("Overview of numerical input attributes after normalization", fontsize=18, fontweight="bold")
pl.tight_layout(pad=3.5)
plt.savefig('Attribute overview after normalization.png', dpi=250)
plt.show()

clean.insert(1, "SqmPrice", data.Price/data.Square)
# Attributes after normalization
plt.figure(figsize=(8, 12))
sns.boxplot(data=clean[['Square', 'LifeSquare', 'KitchenSquare', 'Rooms']], orient='h')
plt.xscale('symlog')
plt.xlim(left=-0.5)
plt.title('Normalized attributes')
plt.savefig('Normalized attributes.png', dpi=250)
# Sqm price after normalization
plt.figure(figsize=(8, 12))
sns.boxplot(data=clean.SqmPrice, orient='h')
plt.title('Price per square meter after data engineering')
plt.savefig('Sqm price after data engineering.png', dpi=250)
plt.show()

clean.drop('SqmPrice', axis=1, inplace=True)

# General information after normalization
print("Info about the dataset (after normalization):", end='\n\n')
print(clean.info(), end='\n\n')
print("General statistic information about attributes (after normalization):", end='\n\n')
print(clean.describe(), end='\n\n')
print("Number of missing values for each column (after normalization):", end='\n\n')
print(clean.isna().sum(), end='\n\n')

# Heatmap
print("Plotting heatmap...")
plt.figure(figsize=(18, 10))
mask = np.triu(np.ones_like(clean.corr(numeric_only=True), dtype=bool))
heatmap = sns.heatmap(clean.corr(numeric_only=True), annot=True, mask=mask, linewidths=0.3, cmap='BrBG', square=False, vmin=-1, vmax=1)
heatmap.set_title('Correlation heatmap', fontsize=18)
plt.tight_layout()
plt.savefig('Heatmap.png', dpi=250)
plt.show()

print("\n------------------------------ TRAINING THE MODEL ------------------------------")

y = clean.Price
X = clean.drop(['Id', 'Price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)

models = {
    'Linear': LinearRegression(),
    'Random forest': RandomForestRegressor(),
    'Tuned Random forest':RandomForestRegressor(max_features=None, max_leaf_nodes=None, min_samples_leaf=1),
    'Histogram Gradient Boosting': HistGradientBoostingRegressor(),
    'Tuned Histogram Gradient Boosting': HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.01, max_leaf_nodes=100),
    'ML Perceptron': MLPRegressor(max_iter=1000),
    'Tuned ML Perceptron': MLPRegressor(activation='relu', hidden_layer_sizes=(512, 256, 128, 64, 32))
}

results = pd.DataFrame(columns=['MAE', 'RMSE', 'R2-score'])

for model_name, model in models.items():

    print("\nTraining a", model_name, "model...")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results.loc[model_name] = [mean_absolute_error(y_test, prediction),
                        mean_squared_error(y_test, prediction, squared=False),
                        r2_score(y_test, prediction)
                        ]

print("\n------------------------------ TRAIN RESULTS ------------------------------")
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'center')
print("\n", results)

# indices = np.argsort(model.feature_importances_)[::-1]

# plt.figure(figsize=(20, 15))
# plt.title("Feature importances", fontsize=16)
# plt.bar(range(X.shape[1]), model.feature_importances_[indices] / model.feature_importances_.sum(),
#         color="#14a1d9", align="center")
# plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, fontsize=10)
# plt.yticks()
# plt.xlim([-1, X.shape[1]])

# plt.show()

# plt.figure(figsize=(16, 8))
# plt.bar(X_train.columns.tolist(), model.feature_importances_)
# plt.show()