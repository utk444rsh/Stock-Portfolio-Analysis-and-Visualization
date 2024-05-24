# Portfolio Optimization Tool

## Overview

This repository contains a portfolio optimization tool that allows users to compare different algorithms for portfolio optimization.

For a detailed description of the algorithms and methodologies used in this tool, refer to [DMPA_ProjRepot[1][1].docx](DMPA_ProjRepot[1][1].docx).

## Usage

To use the portfolio optimization tool, follow these steps:

1. Clone the repository to your local machine:

```
git clone https://github.com/your_username/portfolio-optimization-tool.git
```

2. Install the required dependencies. You can do this by running:

```
pip install -r requirements.txt
```

3. Run the `ult.py` script:

```
python ult.py
```

4. Follow the prompts to input the necessary information, such as the number of portfolios to evaluate and the weights for each stock.

5. The script will compare the results of the three algorithms and display the best-performing algorithm along with the corresponding return percentage.

## File Structure

The repository has the following structure:

```
├── frontier_eff.py           # Implementation of the frontier efficiency algorithm
├── decision_tree.py          # Implementation of the decision tree algorithm
├── kmeans.py                 # Implementation of the K-means algorithm
├── ult.py                    # Main script to compare portfolio optimization algorithms
├── README.md                 # Detailed documentation and instructions
├── requirements.txt          # List of required Python dependencies
└── DMPA_ProjRepot[1][1].docx# Detailed description of algorithms and methodologies
```

## Dependencies

The portfolio optimization tool relies on the following Python packages:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computing.
- `yfinance`: For fetching historical stock data.
- `scikit-learn`: For machine learning algorithms such as decision trees and K-means clustering.
- `matplotlib`: For data visualization.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have any suggestions, enhancements, or bug fixes, please submit a pull request or open an issue.
