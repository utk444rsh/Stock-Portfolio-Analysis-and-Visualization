import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Define the list of stock symbols in your portfolio
stock_symbols = ['AAPL', 'GOOGL', 'NFLX', 'AMZN', 'TSLA']

# Prompt the user to enter the number of portfolios to evaluate
num_user_portfolios = int(input("Enter the number of portfolios to evaluate: "))

# Initialize lists to hold user-defined portfolios and their return percentages
user_weights_list = []
portfolio_returns = []

# Fetch real-time stock data (adjust the date range as needed)
start_date = '2019-01-01'
end_date = '2025-05-05'
stock_data = yf.download(stock_symbols, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns for each stock
returns = stock_data.pct_change().dropna()

# Create a DataFrame to hold the stock returns
stock_returns = pd.DataFrame(data=returns, columns=stock_symbols)

# Loop to input user weights for each portfolio and calculate returns
for _ in range(num_user_portfolios):
    weights = []
    total_weight = 0
    
    print(f"\nEnter weights (as percentages) for each stock in Portfolio {_ + 1}:")
    for symbol in stock_symbols:
        weight = float(input(f"Enter weight of {symbol}: "))
        weights.append(weight)
        total_weight += weight

    # Normalize weights to ensure they sum up to 1
    normalized_weights = [w / total_weight for w in weights]
    user_weights_list.append(normalized_weights)

    # Calculate portfolio return based on the weighted average of stock returns
    portfolio_return = np.sum(normalized_weights * stock_returns.mean()) * 252
    portfolio_returns.append(portfolio_return)

# Define a classification threshold (e.g., 0.0 for binary classification)
threshold = 0.0

# Create a binary target variable (1 for positive returns, 0 for negative returns)
target = (stock_returns > threshold).astype(int)

# Split the data into features (stock returns) and the target variable
X = stock_returns
y = target

# Create and train the ID3 decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Determine the portfolio with the maximum return percentage
max_return_index = np.argmax(portfolio_returns)
max_return = portfolio_returns[max_return_index]
best_portfolio = user_weights_list[max_return_index]

# Print the best portfolio based on return percentage
print(f"\nPortfolio with Maximum Return ({max_return:.2f}%):")
for stock, weight in zip(stock_symbols, best_portfolio):
    if weight > 0:
        print(f"{stock}: {weight * 100:.2f}%")
