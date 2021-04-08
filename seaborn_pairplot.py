from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split

# Create an object of the california housing class
california = fetch_california_housing()

# Set the options for the dataframe
pd.set_option("precision", 4)
pd.set_option("max_columns", 9)
pd.set_option("display.width", None)

# Create a dataframe
california_df = pd.DataFrame(california.data, columns=california.feature_names)

# Add a new column for the target
california_df["MedHouseValue"] = pd.Series(california.target)

# To create a 10% sample from the dataframe
sample_df = california_df.sample(frac=0.1, random_state=17)

# Visualize the features of the dataset
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1)
sns.set_style("whitegrid")

sns.pairplot(data=sample_df, vars=sample_df.columns[0:8])

plt.show()

# Prediction Model
X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state=11
)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# the fit method epects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)

# Use enumerate function to view the coefficients
print("The coefficients are:")
for i, name in enumerate(california.feature_names):
    print(f"{name}: {linear_regression.coef_[i]}")

# Display the intercept of the regression
print("The Intercept is: ", linear_regression.intercept_)

predicted = linear_regression.predict(X_test)
expected = y_test

df = pd.DataFrame()
df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

# Visualize the prediction model
import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(
    data=df, x="Expected", y="Predicted", hue="Predicted", palette="cool", legend=False
)

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)


line = plt2.plot([start, end], [start, end], "k--")

plt2.show()