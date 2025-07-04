import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from time import sleep
from tqdm import tqdm

# --- 配置 ---
# 本地数据存储路径，存储在当前脚本所在目录的上一级下的 'stock_data_akshare' 文件夹
# e.g. /trading/stock_data_akshare/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'stock_data_akshare_hfq')

# 创建存储目录
os.makedirs(DATA_DIR, exist_ok=True)


def get_stock_list():
    """获取当前所有A股列表，并增加重试机制"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print("Getting all A-share stock list from AkShare...")
            # 使用东方财富的接口获取A股列表
            stock_df = ak.stock_zh_a_spot_em()
            print(f"Total stocks: {len(stock_df)}")
            return stock_df['代码'].tolist()
        except Exception as e:
            print(f"Error getting stock list on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                sleep(5)  # Wait for 5 seconds before retrying
            else:
                print("Failed to get stock list after multiple retries.")
    return []


def update_single_stock_data(stock_code: str):
    """下载或更新单个股票的后复权日线数据。返回一个包含状态和信息的字典。"""
    filepath = os.path.join(DATA_DIR, f"{stock_code}.csv")
    
    # 设定一个较早的日期作为首次下载的起始日，确保能获取到10年数据
    start_date = '20140101'

    # 如果文件已存在，则进行增量更新
    if os.path.exists(filepath):
        try:
            df_local = pd.read_csv(filepath)
            if not df_local.empty and 'Date' in df_local.columns:
                last_date_str = df_local['Date'].iloc[-1]
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
        except pd.errors.EmptyDataError:
            # 如果文件是空的，当作新文件处理
            pass

    today = datetime.now().strftime('%Y%m%d')
    if start_date > today:
        return {'status': 'skipped', 'code': stock_code, 'message': 'Already up to date.'}

    try:
        # 使用 ak.stock_zh_a_hist 获取后复权（hfq）数据
        df_new = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=today, adjust="hfq")
        
        if df_new is None or df_new.empty:
            return {'status': 'no_data', 'code': stock_code, 'message': 'No new data found from provider.'}

        # 数据清洗和重命名
        df_new.rename(columns={'日期': 'Date', '开盘': 'Open', '最高': 'High',
                               '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'}, inplace=True)
        
        # 仅保留需要的列
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df_new = df_new[required_columns]
        
        # 将日期列转换为datetime对象，然后再格式化为字符串，以避免linter错误
        date_series = pd.to_datetime(df_new['Date'])
        df_new['Date'] = date_series.dt.strftime('%Y-%m-%d')
        
        # 存储数据
        if os.path.exists(filepath):
            # 增量更新模式
            df_new.to_csv(filepath, mode='a', header=False, index=False)
        else:
            # 首次下载模式
            df_new.to_csv(filepath, index=False)
        
        return {'status': 'success', 'code': stock_code, 'message': f"Updated from {start_date} to {today}."}

    except Exception as e:
        return {'status': 'error', 'code': stock_code, 'message': str(e)}


def main_parallel():
    """主程序，使用并行处理来下载和更新数据"""
    stock_list = get_stock_list()
    if not stock_list:
        print("Could not retrieve stock list. Exiting.")
        return

    # 设置进程数
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for data synchronization...")

    success_count = 0
    no_data_count = 0
    error_count = 0
    skipped_count = 0
    failed_stocks = []

    # 创建进程池
    with Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 来获得更好的进度条体验
        results_iterator = pool.imap_unordered(update_single_stock_data, stock_list)
        
        # 使用 tqdm 显示进度
        for result in tqdm(results_iterator, total=len(stock_list), desc="Syncing Data"):
            status = result.get('status', 'error')
            if status == 'success':
                success_count += 1
            elif status == 'no_data':
                no_data_count += 1
                failed_stocks.append(result)
            elif status == 'error':
                error_count += 1
                failed_stocks.append(result)
            elif status == 'skipped':
                skipped_count += 1
            
    print("\nData synchronization finished!")
    print("-" * 30)
    print(f"Total stocks in list:      {len(stock_list)}")
    print(f"Successfully downloaded:   {success_count}")
    print(f"Skipped (already up-to-date): {skipped_count}")
    print(f"No data from provider:     {no_data_count}")
    print(f"Errors during fetching:      {error_count}")
    print("-" * 30)

    if failed_stocks:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'failed_stocks.log')
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("Stock Code,Status,Message\n")
                for stock in failed_stocks:
                    # 将可能包含逗号的消息内容用引号包裹起来
                    message = str(stock['message']).replace('"', '""')
                    f.write(f"{stock['code']},{stock['status']},\"{message}\"\n")
            print(f"Details for stocks that couldn't be fetched are saved in: {log_file}")
        except IOError as e:
            print(f"Error writing to log file {log_file}: {e}")
            
    print(f"All data saved in: {DATA_DIR}")


if __name__ == '__main__':
    main_parallel() 