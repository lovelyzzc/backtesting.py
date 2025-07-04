import pandas as pd
import numpy as np

def analyze_backtest_results(filepath):
    """
    Analyzes the backtesting results from a CSV file.

    Args:
        filepath (str): The path to the CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return

    # --- Data Cleaning and Preparation ---
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert percentage columns to numeric, handling potential errors
    for col in ['Exposure Time [%]', 'Return [%]', 'Buy & Hold Return [%]', 
                'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 'CAGR [%]', 
                'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 
                'Expectancy [%]']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Analysis ---
    print("--- 回测结果整体分析 ---")
    
    # Calculate and print key metrics
    print("\n### 关键指标平均值:")
    key_metrics_avg = {
        '年化回报率 (Return Ann. %)': df['Return (Ann.) [%]'].mean(),
        '夏普比率 (Sharpe Ratio)': df['Sharpe Ratio'].mean(),
        '胜率 (Win Rate %)': df['Win Rate [%]'].mean(),
        '最大回撤 (Max. Drawdown %)': df['Max. Drawdown [%]'].mean(),
        '盈利因子 (Profit Factor)': df['Profit Factor'].mean(),
        '交易次数 (# Trades)': df['# Trades'].mean()
    }
    for name, value in key_metrics_avg.items():
        print(f"- {name}: {value:.2f}")

    print("\n### 总体统计描述:")
    # Using a relevant subset of columns for describe() to keep it concise
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 15)
    print(df[['Return (Ann.) [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Max. Drawdown [%]', 'Profit Factor', '# Trades']].describe())

    # --- Top/Bottom Performers ---
    print("\n--- 表现最佳/最差的策略 ---")
    
    # Sort by Sharpe Ratio and Annual Return for a more robust "best" list
    df_cleaned = df.dropna(subset=['Sharpe Ratio', 'Return (Ann.) [%]'])
    
    print("\n### 夏普比率排名前5的股票:")
    top_5_sharpe = df_cleaned.sort_values(by='Sharpe Ratio', ascending=False).head(5)
    print(top_5_sharpe[['Stock', 'Sharpe Ratio', 'Return (Ann.) [%]', 'Win Rate [%]', 'Max. Drawdown [%]']])
    
    print("\n### 年化回报率排名前5的股票:")
    top_5_return = df_cleaned.sort_values(by='Return (Ann.) [%]', ascending=False).head(5)
    print(top_5_return[['Stock', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Max. Drawdown [%]']])
    
    print("\n### 年化回报率排名后5的股票:")
    bottom_5_return = df_cleaned.sort_values(by='Return (Ann.) [%]', ascending=True).head(5)
    print(bottom_5_return[['Stock', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Win Rate [%]', 'Max. Drawdown [%]']])
    
    print("\n### 最大回撤最大的5个策略:")
    max_drawdown = df_cleaned.sort_values(by='Max. Drawdown [%]', ascending=True).head(5)
    print(max_drawdown[['Stock', 'Max. Drawdown [%]', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Win Rate [%]']])

def portfolio_analysis(returns_df):
    """
    Performs portfolio-level analysis.

    Args:
        returns_df (pd.DataFrame): DataFrame with daily returns for each stock.
                                   Index should be datetime, columns are stock tickers.
    """
    print("\n\n--- 组合表现分析 ---")

    # Assuming equal weight for simplicity
    num_assets = len(returns_df.columns)
    weights = np.array([1/num_assets] * num_assets)

    # Calculate daily portfolio returns
    portfolio_returns = returns_df.mean(axis=1) # Simple average for equal weight

    # Calculate cumulative portfolio returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # --- Key Portfolio Metrics ---
    print("\n### 组合关键指标:")
    
    total_days = len(portfolio_returns)
    cagr = (cumulative_returns.iloc[-1])**(252/total_days) - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = cagr / annual_volatility # Assuming risk-free rate is 0

    # Max Drawdown
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    print(f"- 年化复合增长率 (CAGR): {cagr:.2%}")
    print(f"- 年化波动率: {annual_volatility:.2%}")
    print(f"- 夏普比率: {sharpe_ratio:.2f}")
    print(f"- 最大回撤: {max_drawdown:.2%}")

    # Plotting logic would go here
    # import matplotlib.pyplot as plt
    # cumulative_returns.plot(title='Portfolio Cumulative Returns')
    # plt.show()


if __name__ == "__main__":
    analyze_backtest_results('momentum_trail_backtest_results.csv')
    
    # --- Example of how to call portfolio analysis ---
    # This part is commented out as we don't have the daily returns data yet.
    # try:
    #     # Assuming you have a CSV with daily returns for each stock
    #     # e.g., columns: Date, STOCK1_RETURN, STOCK2_RETURN, ...
    #     all_returns = pd.read_csv('daily_returns.csv', index_col='Date', parse_dates=True)
    #     portfolio_analysis(all_returns)
    # except FileNotFoundError:
    #     print("\n'daily_returns.csv' not found. Skipping portfolio analysis.")
    #     print("Please generate this file from your backtesting script.") 