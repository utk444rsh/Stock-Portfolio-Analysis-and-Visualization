# ult.py
from frontier_eff import calculate_max_return
from decision_tree import calculate_max_return as calculate_max_return_2

def compare_portfolio_results():
    max_return_prog1 = calculate_max_return()
    max_return_prog2 = calculate_max_return_2()

if __name__ == "__main__":
    compare_portfolio_results()
