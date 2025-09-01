# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from backtesting import Backtest
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.reverse_rsi.strategies import ReverseRsiLongOnlyStrategy

def analyze_stock(filename, data_dir, start_date, end_date):
    """
    分析单个股票文件，检测最后一天的买入或卖出信号
    这是为并行处理设计的工作函数
    """
    filepath = os.path.join(data_dir, filename)
    stock_symbol = os.path.splitext(filename)[0]
    
    try:
        # 从Tushare CSV加载数据
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        df = df.sort_index()

        # backtesting.py需要列名: 'Open', 'High', 'Low', 'Close', 'Volume'
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'vol': 'Volume'
        }, inplace=True)

        # 过滤相关时期的数据，确保不包含未来日期
        df_filtered = df.loc[start_date:end_date]
        
        # 额外检查：确保不包含今天之后的数据
        today = pd.Timestamp(datetime.now().date())
        if df_filtered.index[-1] > today:
            df_filtered = df_filtered[df_filtered.index <= today]

        # 策略需要足够的数据来生成指标
        if len(df_filtered) < 100:  # Reverse RSI策略需要足够数据
            return None

        # 运行回测，设置足够的初始资金以避免价格超出警告
        bt = Backtest(df_filtered, ReverseRsiLongOnlyStrategy, cash=1000000, commission=0.0)
        stats = bt.run()

        # 检查最后一天的信号
        strategy_instance = stats._strategy
        
        # 获取最后几天的数据来检测信号
        if len(strategy_instance.data.Close) >= 2:
            current_price = strategy_instance.data.Close[-1]
            
            # 检查各种信号条件
            buy_signal = False
            sell_signal = False
            signal_type = ""
            
            # 1. 检查价格突破信号
            if (hasattr(strategy_instance, 'os_price') and 
                len(strategy_instance.os_price) >= 2 and
                hasattr(strategy_instance, 'ob_price') and 
                len(strategy_instance.ob_price) >= 2):
                
                # 买入：价格突破超卖水平向上
                if (current_price > strategy_instance.os_price[-1] and 
                    strategy_instance.data.Close[-2] <= strategy_instance.os_price[-2]):
                    buy_signal = True
                    signal_type += "突破超卖+"
                
                # 卖出：价格突破超买水平向下
                if (current_price < strategy_instance.ob_price[-1] and
                    strategy_instance.data.Close[-2] >= strategy_instance.ob_price[-2]):
                    sell_signal = True
                    signal_type += "突破超买+"
            
            # 2. 检查发散信号
            if (hasattr(strategy_instance, 'bull_divergence') and 
                len(strategy_instance.bull_divergence) > 0):
                if strategy_instance.bull_divergence[-1]:
                    buy_signal = True
                    signal_type += "看涨发散+"
                    
            if (hasattr(strategy_instance, 'bear_divergence') and 
                len(strategy_instance.bear_divergence) > 0):
                if strategy_instance.bear_divergence[-1]:
                    sell_signal = True
                    signal_type += "看跌发散+"
            
            # 3. 检查SuperTrend趋势转换
            if (hasattr(strategy_instance, 'st_direction') and 
                len(strategy_instance.st_direction) >= 2):
                
                # SuperTrend转为看涨 (从-1转为1)
                if (strategy_instance.st_direction[-1] == 1 and 
                    strategy_instance.st_direction[-2] == -1):
                    buy_signal = True
                    signal_type += "趋势转涨+"
                
                # SuperTrend转为看跌 (从1转为-1)
                if (strategy_instance.st_direction[-1] == -1 and 
                    strategy_instance.st_direction[-2] == 1):
                    sell_signal = True
                    signal_type += "趋势转跌+"
            
            # 趋势过滤：只有在SuperTrend确认的情况下才报告信号
            if (hasattr(strategy_instance, 'st_direction') and 
                len(strategy_instance.st_direction) > 0):
                
                if buy_signal and strategy_instance.st_direction[-1] == 1:
                    print(f"✅ Reverse RSI买入信号: {stock_symbol} - {signal_type.rstrip('+')}")
                    return (stock_symbol, 'buy', signal_type.rstrip('+'))
                elif sell_signal and strategy_instance.st_direction[-1] == -1:
                    # print(f"❌ Reverse RSI卖出信号: {stock_symbol} - {signal_type.rstrip('+')}")
                    return (stock_symbol, 'sell', signal_type.rstrip('+'))

    except Exception as e:
        # 静默忽略个别股票的错误，避免中断整个扫描
        # 更强大的解决方案可以将这些错误记录到单独的文件中
        print(f"⚠️ 分析 {stock_symbol} 时出错: {str(e)}")
        return None
    
    return None

def find_reverse_rsi_signals(data_dir, start_date='2020-01-01', end_date='2025-07-08'):
    """
    并行扫描目录中的所有CSV文件，寻找最后一天有买入或卖出信号的股票
    """
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    n_cores = cpu_count()
    print(f"发现 {len(files)} 个CSV文件。开始使用 {n_cores} 个CPU核心分析...")

    # 使用functools.partial创建一个预填充了部分参数的新函数
    # 这是在pool.map中向工作函数传递额外参数的简洁方式
    process_func = partial(analyze_stock, 
                           data_dir=data_dir, 
                           start_date=start_date, 
                           end_date=end_date)

    # 进程池管理一组工作进程
    with Pool(n_cores) as p:
        # map将process_func应用到'files'列表中的每个项目
        # 处理过程在工作进程之间分布
        results = p.map(process_func, files)
    
    # 过滤掉None值并分离信号
    valid_results = [res for res in results if res is not None]
    buy_signal_stocks = [(stock, signal_desc) for stock, sig_type, signal_desc in valid_results if sig_type == 'buy']
    sell_signal_stocks = [(stock, signal_desc) for stock, sig_type, signal_desc in valid_results if sig_type == 'sell']
    
    return buy_signal_stocks, sell_signal_stocks

def save_signals_to_ini(filepath, stock_list, block_name):
    """将股票代码列表保存到同花顺的INI文件中"""
    if stock_list:
        print(f"\n--- 发现 {block_name} 信号 ---")
        
        formatted_stocks = []
        for stock_info in stock_list:
            try:
                if isinstance(stock_info, tuple):
                    stock_symbol, signal_desc = stock_info
                else:
                    stock_symbol = stock_info
                    signal_desc = ""
                
                code = stock_symbol.split('.')[0]
                formatted_stocks.append(code)
                print(f"- {stock_symbol} -> {code} ({signal_desc})")
            except (IndexError, ValueError):
                print(f"  - 警告: 跳过格式错误的股票代码 {stock_symbol}")
                continue
        
        with open(filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write(f"股票代码={','.join(formatted_stocks)}\n")
            f.write(f"板块名称={block_name}\n")

        print(f"\n同花顺导入文件已保存到: {filepath}")
    else:
        print(f"\n--- 未发现 {block_name} 信号 ---")
        # 如果没有找到股票，创建一个空的INI文件
        with open(filepath, 'w', encoding='gbk') as f:
            f.write("[自选股设置]\n")
            f.write("股票代码=\n")
            f.write(f"板块名称={block_name}\n")
        print(f"\n已创建空结果文件: {filepath}")

if __name__ == "__main__":
    # 定义数据目录 (请确保路径正确)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')

    # 运行扫描
    # 使用合理的结束日期，避免未来数据
    end_date = '2024-12-31'  # 或者使用实际的最新交易日
    print("开始Reverse RSI策略股票扫描...")
    
    buy_stocks, sell_stocks = find_reverse_rsi_signals(
        data_dir=data_dir,
        start_date='2024-01-01',
        end_date=end_date
    )

    # 打印并保存结果
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存买入信号
    buy_output_filepath = os.path.join(results_dir, 'Reverse_RSI_买入信号.ini')
    save_signals_to_ini(buy_output_filepath, buy_stocks, "Reverse RSI买入")

    # 保存卖出信号
    sell_output_filepath = os.path.join(results_dir, 'Reverse_RSI_卖出信号.ini')
    save_signals_to_ini(sell_output_filepath, sell_stocks, "Reverse RSI卖出")
    
    print(f"\n🎉 扫描完成！结果已保存到 results 目录") 