import os
import time
from datetime import datetime

import pandas as pd
import tushare as ts

# --- 配置 ---
# 从环境变量中获取 Tushare Token
TUSHARE_TOKEN = "20ac0229db965307a457e8c4573be0df7f5e36b29b8a166803ad446c"
if not TUSHARE_TOKEN:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

# 数据存储路径

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '..', 'tushare_data')
DAILY_DATA_PATH = os.path.join(DATA_PATH, "daily")
STOCK_BASIC_FILE = os.path.join(DATA_PATH, "all_stocks.csv")

# 创建数据存储目录
os.makedirs(DAILY_DATA_PATH, exist_ok=True)

# 初始化 Tushare Pro API
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def sync_all_stock_basic():
    """
    同步所有A股基本信息
    """
    print("正在同步所有A股基本信息...")
    try:
        data = pro.stock_basic(
            exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date"
        )
        data.to_csv(STOCK_BASIC_FILE, index=False)
        print(f"A股基本信息已保存至 {STOCK_BASIC_FILE}")
        return data
    except Exception as e:
        print(f"同步A股基本信息失败: {e}")
        return None


def sync_all_daily_data(start_date="20200101", end_date=None):
    """
    同步所有A股的日线数据，支持增量更新。
    """
    if not os.path.exists(STOCK_BASIC_FILE):
        print(f"未找到股票基本信息文件: {STOCK_BASIC_FILE}")
        print("请先运行 sync_all_stock_basic() 函数同步基本信息。")
        stock_basic = sync_all_stock_basic()
        if stock_basic is None:
            return
    else:
        stock_basic = pd.read_csv(STOCK_BASIC_FILE)

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    total_stocks = len(stock_basic)
    print(f"总共需要同步 {total_stocks} 只股票的日线数据...")

    for i in range(len(stock_basic)):
        row = stock_basic.iloc[i]
        ts_code = str(row["ts_code"])
        file_path = os.path.join(DAILY_DATA_PATH, f"{ts_code}.csv")
        
        print(f"[{i + 1}/{total_stocks}] 正在处理 {ts_code} ({row['name']})...")

        # --- 增量更新逻辑 ---
        sync_start_date = start_date
        is_incremental = False
        if os.path.exists(file_path):
            try:
                # 读取本地数据最后一条记录的日期
                # 为了效率，可以只读取最后一行的日期，但直接用pandas更简单鲁棒
                # 不使用 usecols 来规避 linter 的类型检查问题
                existing_df = pd.read_csv(file_path)
                if not existing_df.empty:
                    last_date = pd.to_datetime(existing_df['Date'].iloc[-1])
                    # 新的开始日期是本地最后日期的下一天
                    sync_start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y%m%d')
                    is_incremental = True
            except Exception as e:
                print(f"  -> 读取本地数据 {file_path} 失败: {e}。将执行全量同步。")
        
        mode_str = "增量" if is_incremental else "全量"
        print(f"  -> 模式: {mode_str}同步, 起始日期: {sync_start_date}")

        try:
            # 如果计算出的起始日期大于结束日期，说明数据已经是最新
            if sync_start_date > end_date:
                print(f"  -> {ts_code} 数据已是最新，无需更新。")
                continue

            # 使用 tushare.pro_bar 获取后复权日线数据
            df = ts.pro_bar(
                ts_code=ts_code,
                adj="hfq",
                start_date=sync_start_date,
                end_date=end_date,
            )
            
            if df is not None and not df.empty:
                # --- 数据格式化以适配 backtesting.py ---
                df.rename(columns={
                    'trade_date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'vol': 'Volume'
                }, inplace=True)

                # 将 'Date' 列转换为 datetime 对象，明确格式以提高稳定性
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

                # 按日期升序排序
                df.sort_values('Date', inplace=True)

                # 将 'Date' 列设为索引
                df.set_index('Date', inplace=True)
                
                # 选择 backtesting.py 需要的列
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

                if is_incremental:
                    df.to_csv(file_path, mode='a', header=False)
                    print(f"  -> {ts_code} 新数据已追加至 {file_path}")
                else:
                    df.to_csv(file_path)
                    print(f"  -> {ts_code} 数据已保存至 {file_path}")
            else:
                print(f"  -> 未获取到 {ts_code} 在 {sync_start_date} 之后的新数据。")

            # tushare pro 免费版有积分限制，每分钟不能超过一定次数的调用
            time.sleep(0.12)

        except Exception as e:
            print(f"  -> 同步 {ts_code} 数据时出错: {e}")
            # 即使出错也继续尝试下一只股票
            continue


if __name__ == "__main__":
    print("开始执行数据同步任务...")
    
    # 步骤1: 同步股票基本信息
    sync_all_stock_basic()
    
    # 步骤2: 同步所有股票的日线数据
    # 您可以根据需要修改这里的起止日期
    sync_all_daily_data(start_date="20200101")
    
    print("所有数据同步任务完成！")
