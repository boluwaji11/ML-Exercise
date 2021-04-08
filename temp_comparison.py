import pandas as pd

nyc_average_yearly = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")
nyc_january = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

# Floor the Date values for the average yearly temp
nyc_average_yearly.Date = nyc_average_yearly.Date.floordiv(100)

# Split the dataset into training and test data for the january and yearly average temperaturs
from sklearn.model_selection import train_test_split

# Average Yearly
X_train, X_test, y_train, y_test = train_test_split(
    nyc_average_yearly.Date.values.reshape(-1, 1),
    nyc_average_yearly.Value.values,
    random_state=11,
)

# January Average
X_jan_train, X_jan_test, y_jan_train, y_jan_test = train_test_split(
    nyc_january.Date.values.reshape(-1, 1),
    nyc_january.Temperature.values,
    random_state=11,
)


# Run the linear regression
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression_jan = LinearRegression()

# the fit method epects the samples and the targets for training
linear_regression.fit(X=X_train, y=y_train)
linear_regression_jan.fit(X=X_jan_train, y=y_jan_train)

predicted = linear_regression.predict(X_test)
expected = y_test

predicted_jan = linear_regression_jan.predict(X_jan_test)
expected_jan = y_jan_test


# lambda implements y=mx+b
predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_
predict_jan = (
    lambda x: linear_regression_jan.coef_ * x + linear_regression_jan.intercept_
)

# Visualize the two scatter plots
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, figsize=(10, 6))

fig.suptitle(
    "1895-2018: Comparing Average Yearly Temperature and Average Temperature in January",
    fontsize=14,
)

ax[0].scatter(
    x=nyc_average_yearly["Date"], y=nyc_average_yearly["Value"], c="darkgreen"
)
ax[0].set_xlabel("Year")
ax[0].set_ylabel("Average Yearly Temperature")
ax[0].set_ylim(10, 70)

ax[1].scatter(x=nyc_january["Date"], y=nyc_january["Temperature"], c="darkgreen")
ax[1].set_xlabel("Year")
ax[1].set_ylabel("Average January Temperature")

plt.ylim(10, 70)

# Add the trend lines
import numpy as np

x = np.array([min(nyc_average_yearly.Date.values), max(nyc_average_yearly.Date.values)])
y = predict(x)

x_jan = np.array([min(nyc_january.Date.values), max(nyc_january.Date.values)])
y_jan = predict_jan(x_jan)

line = ax[0].plot(x, y, c="mediumseagreen", linewidth=2)
line2 = ax[1].plot(x_jan, y_jan, c="mediumseagreen", linewidth=2)

plt.show()
