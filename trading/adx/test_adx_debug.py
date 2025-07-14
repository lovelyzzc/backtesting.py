# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import traceback

# --- Setup Python Path to allow imports from parent directory ---
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading.adx.indicators import adx_indicator

def test_adx_with_real_data():
    """使用真实数据测试ADX指标"""
    try:
        # 读取一个真实的CSV文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'tushare_data', 'daily')
        csv_file = os.path.join(data_dir, '000001.SZ.csv')
        
        print(f"读取数据文件: {csv_file}")
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        
        print(f"数据形状: {df.shape}")
        print("前5行数据:")
        print(df.head())
        print("\n数据类型:")
        print(df.dtypes)
        
        # 检查是否有NaN值
        print(f"\n缺失值检查:")
        print(df.isnull().sum())
        
        # 使用不同的长度测试ADX指标
        lengths_to_test = [10, 14, 15, 20]
        
        for length in lengths_to_test:
            print(f"\n=== 测试 ADX 长度 {length} ===")
            try:
                # 确保数据是pandas Series类型
                high_series = pd.Series(df['High'])
                low_series = pd.Series(df['Low'])
                close_series = pd.Series(df['Close'])
                
                adx, di_plus, di_minus = adx_indicator(
                    high_series, low_series, close_series, length
                )
                
                print(f"ADX 计算成功!")
                print(f"ADX 形状: {adx.shape}")
                print(f"ADX 前10个值: {adx[:10]}")
                print(f"ADX 最后10个值: {adx[-10:]}")
                
                # 检查结果中的NaN和无穷大值
                adx_nan_count = np.isnan(adx).sum()
                adx_inf_count = np.isinf(adx).sum()
                print(f"ADX 中的 NaN 数量: {adx_nan_count}")
                print(f"ADX 中的 无穷大 数量: {adx_inf_count}")
                
                di_plus_nan_count = np.isnan(di_plus).sum()
                di_minus_nan_count = np.isnan(di_minus).sum()
                print(f"DI+ 中的 NaN 数量: {di_plus_nan_count}")
                print(f"DI- 中的 NaN 数量: {di_minus_nan_count}")
                
            except Exception as e:
                print(f"ADX 计算失败: {e}")
                print("详细错误信息:")
                traceback.print_exc()
                
    except Exception as e:
        print(f"读取数据或设置失败: {e}")
        traceback.print_exc()

def test_adx_with_synthetic_data():
    """使用合成数据测试ADX指标"""
    print("\n=== 使用合成数据测试 ===")
    
    # 创建一些合成的价格数据
    n_periods = 100
    np.random.seed(42)
    
    # 生成价格序列
    base_price = 100
    price_changes = np.random.normal(0, 1, n_periods)
    close_prices = [base_price]
    
    for change in price_changes[1:]:
        close_prices.append(close_prices[-1] + change)
    
    close_prices = np.array(close_prices)
    high_prices = close_prices + np.random.uniform(0, 2, n_periods)
    low_prices = close_prices - np.random.uniform(0, 2, n_periods)
    
    # 创建pandas Series
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)  
    close_series = pd.Series(close_prices)
    
    print(f"合成数据长度: {len(close_series)}")
    print(f"High 范围: {high_series.min():.2f} - {high_series.max():.2f}")
    print(f"Low 范围: {low_series.min():.2f} - {low_series.max():.2f}")
    print(f"Close 范围: {close_series.min():.2f} - {close_series.max():.2f}")
    
    try:
        adx, di_plus, di_minus = adx_indicator(high_series, low_series, close_series, 14)
        print("合成数据 ADX 计算成功!")
        print(f"ADX 最后5个值: {adx[-5:]}")
        print(f"DI+ 最后5个值: {di_plus[-5:]}")
        print(f"DI- 最后5个值: {di_minus[-5:]}")
    except Exception as e:
        print(f"合成数据 ADX 计算失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("开始 ADX 指标调试测试...")
    test_adx_with_real_data()
    test_adx_with_synthetic_data()
    print("\n测试完成!") 