import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- 配置 ---
# 获取当前脚本所在的目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 源数据目录 (由 sync_akshare_data.py 下载)
SOURCE_DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'stock_data_akshare_hfq')

# 清洗后数据的存储目录
CLEANED_DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'stock_data_cleaned')

# 创建存储目录
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)


def clean_single_stock_file(filename: str):
    """
    清洗单个股票的CSV文件。
    
    清洗步骤:
    1. 读取数据。
    2. 转换日期列为 datetime 对象。
    3. 按日期排序并移除重复的日期行。
    4. 处理价格为0或负数的异常数据行。
    5. 处理缺失值 (NaN)。
    6. 保存清洗后的数据。
    
    :param filename: 不带路径的文件名, e.g., "000001.csv"
    :return: 处理结果的字符串消息。
    """
    source_filepath = os.path.join(SOURCE_DATA_DIR, filename)
    cleaned_filepath = os.path.join(CLEANED_DATA_DIR, filename)
    
    stock_code = filename.split('.')[0]

    if not os.path.exists(source_filepath):
        return f"{stock_code}: Source file not found."

    try:
        # 1. 读取数据
        df = pd.read_csv(source_filepath)

        if df.empty:
            return f"{stock_code}: Source file is empty."
            
        # 记录原始行数
        original_rows = len(df)

        # 2. 转换日期列为datetime对象
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 3. 按日期排序并移除重复项
        df.sort_values(by='Date', inplace=True)
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)

        # 4. 移除价格 <= 0 的异常行 (通常是数据错误)
        price_cols = ['Open', 'High', 'Low', 'Close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        # 5. 处理缺失值 (NaN)
        # 使用前向填充，因为当天的价格最可能与前一交易日相似
        df.ffill(inplace=True)
        # 如果第一行就有NaN，前向填充无效，再用后向填充处理
        df.bfill(inplace=True)

        # 如果填充后仍有NaN (可能整个文件都是NaN)，则放弃该文件
        if df.isnull().values.any():
            return f"{stock_code}: Contains persistent NaN values after cleaning, skipped."
            
        # 格式化日期列为字符串以便存储
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # 记录清洗后行数
        cleaned_rows = len(df)

        # 6. 保存清洗后的文件
        df.to_csv(cleaned_filepath, index=False)
        
        return f"{stock_code}: Cleaned. Rows: {original_rows} -> {cleaned_rows}."

    except Exception as e:
        return f"{stock_code}: Error cleaning file - {e}"


def main_parallel_cleaning():
    """主程序，使用并行处理来清洗所有股票数据"""
    
    # 检查源目录是否存在
    if not os.path.isdir(SOURCE_DATA_DIR):
        print(f"Source data directory not found: {SOURCE_DATA_DIR}")
        print("Please run 'sync_akshare_data.py' first to download data.")
        return

    # 获取所有需要清洗的股票文件列表
    stock_files = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith('.csv')]
    
    if not stock_files:
        print(f"No CSV files found in {SOURCE_DATA_DIR}. Nothing to clean.")
        return

    # 设置进程数
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for data cleaning...")
    print(f"Found {len(stock_files)} files to clean.")

    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 来获得更好的进度条体验
        results_iterator = pool.imap_unordered(clean_single_stock_file, stock_files)
        
        # 使用 tqdm 显示进度
        for result in tqdm(results_iterator, total=len(stock_files), desc="Cleaning Data"):
            # 可以选择性地打印每个文件的处理结果，但为了进度条整洁，这里注释掉
            # print(result)
            pass
            
    print("\nData cleaning finished!")
    print(f"All cleaned data saved in: {CLEANED_DATA_DIR}")


if __name__ == '__main__':
    main_parallel_cleaning() 