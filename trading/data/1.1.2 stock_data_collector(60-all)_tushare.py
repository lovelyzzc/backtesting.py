import tushare as ts
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import logging
from pathlib import Path
import backoff
import random
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataCollector:
    def __init__(self, db_name='stock_60m_tushare.db'):
        self.db_name = db_name
        self.conn: Optional[sqlite3.Connection] = None
        self.ts_api = None
        # tushare 需要的时间格式是 YYYY-MM-DD HH:MM:SS
        # 与日线数据对齐：使用960天的时间周期
        self.end_date = datetime.now().strftime('%Y-%m-%d 15:00:00')
        self.start_date = (datetime.now() - timedelta(days=960)).strftime('%Y-%m-%d 09:30:00')

    def initialize(self):
        """初始化连接"""
        try:
            # 使用您的 tushare token
            self.ts_api = ts.pro_api('20ac0229db965307a457e8c4573be0df7f5e36b29b8a166803ad446c')
            self.conn = sqlite3.connect(self.db_name)
            logger.info("初始化成功")
            logger.info(f"数据时间范围: {self.start_date} 至 {self.end_date}")
            logger.info("时间周期已对齐日线数据：960天 (约2.6年)")

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()

    def get_processed_stocks(self):
        """获取已处理的股票列表"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            processed_stocks = [table[0].replace('_', '.') for table in tables]
            return processed_stocks
        except Exception as e:
            logger.error(f"获取已处理股票列表失败: {e}")
            return []

    def filter_stock_code(self, code):
        """过滤特定股票代码"""
        if code.startswith('688'):  # 过滤科创板
            return False
        if code.startswith('bj'):  # 过滤北交所
            return False
        return True

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=120)
    def get_stock_list(self):
        """获取A股股票列表"""
        try:
            time.sleep(0.6)  # 防止触发限流
            stock_list = self.ts_api.stock_basic(exchange='', list_status='L')
            filtered_stocks = []

            for _, row in stock_list.iterrows():
                if self.filter_stock_code(row['symbol']):
                    # 转换为tushare格式的代码
                    if row['symbol'].startswith('6'):
                        code = f"{row['symbol']}.SH"
                    else:
                        code = f"{row['symbol']}.SZ"
                    filtered_stocks.append(code)

            logger.info(f"过滤后的股票数量: {len(filtered_stocks)}")
            return filtered_stocks
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60)
    def query_stock_data(self, stock_code):
        """查询单个股票的60分钟数据"""
        try:
            time.sleep(random.uniform(10, 15))  # 调整API调用间隔，提高效率

            all_data = []
            current_end_date = self.end_date
            
            # 分批获取数据，每次最多获取8000行
            while True:
                # 使用 tushare 的 stk_mins 接口获取60分钟数据
                df = self.ts_api.stk_mins(
                    ts_code=stock_code,
                    freq='60min',  # 60分钟频度
                    start_date=self.start_date,
                    end_date=current_end_date
                )

                if df is None or df.empty:
                    break
                
                # 如果数据量少于8000行，说明已经获取完所有数据
                if len(df) < 8000:
                    all_data.append(df)
                    break
                
                # 如果数据量等于8000行，需要继续获取更早的数据
                all_data.append(df)
                
                # 更新结束时间为当前批次最早的时间
                earliest_time = df['trade_time'].min()
                current_end_date = pd.to_datetime(earliest_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # 防止无限循环
                if pd.to_datetime(current_end_date) <= pd.to_datetime(self.start_date):
                    break
                
                # 批次间暂停，避免API限流
                time.sleep(random.uniform(2, 5))
            
            # 合并所有数据
            if not all_data:
                return None
                
            df = pd.concat(all_data, ignore_index=True)
            
            # 去重（以防重复数据）
            df = df.drop_duplicates(subset=['trade_time'], keep='first')

            # 转换数据格式以匹配原程序的结构
            df = df.rename(columns={
                'ts_code': 'code',
                'trade_time': 'datetime',
                'vol': 'volume',
                'amount': 'amount'
            })

            # 处理时间格式：将 trade_time 拆分为 date 和 time
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
            df['time'] = df['datetime'].dt.strftime('%H:%M')

            # 重新排列列的顺序，保持与原程序一致
            df = df[['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']]

            logger.info(f"{stock_code} 获取数据 {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"查询股票 {stock_code} 数据失败: {e}")
            return None

    def test_data_format(self, stock_code):
        """测试数据格式"""
        try:
            df = self.query_stock_data(stock_code)
            if df is None or df.empty:
                logger.warning(f"{stock_code} 无测试数据")
                return False

            # 创建datetime列用于排序
            df['datetime_full'] = pd.to_datetime(df['date'] + ' ' + df['time'])

            # 显示5条测试数据
            print("\n=== Tushare 60分钟数据格式测试 ===")
            print(f"测试股票: {stock_code}")
            print("最新5条记录：")
            test_df = df.sort_values('datetime_full', ascending=False).head(5)
            display_df = test_df[['date', 'time', 'open', 'close', 'high', 'low', 'volume']].copy()
            print(display_df.to_string(index=False))
            print(f"总记录数: {len(df)}")
            print("=" * 50)

            return True

        except Exception as e:
            logger.error(f"测试数据格式失败: {e}")
            return False

    def process_single_stock(self, stock_code):
        """处理单只股票数据"""
        try:
            df = self.query_stock_data(stock_code)

            if df is None or df.empty:
                logger.warning(f"{stock_code} 无数据")
                return None

            # 创建完整的datetime列用于排序和处理
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

            # 60分钟K线数据处理
            df = df.sort_values('datetime', ascending=False).head(960)  # 保持数据量不变
            df = df.sort_values('datetime')

            # 确保数值列的类型正确
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df[df[col].notna()]

            if df.empty:
                logger.warning(f"{stock_code} 数值处理后数据为空")
                return None

            # 计算技术指标
            df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['DIF'] = df['EMA12'] - df['EMA26']
            df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
            df['MACD'] = 2 * (df['DIF'] - df['DEA'])

            df = df.sort_values('datetime', ascending=False)

            if len(df) < 10:
                logger.warning(f"{stock_code} 数据量过少")
                return None

            # 格式化datetime为字符串格式
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

            # 保存到数据库
            table_name = stock_code.replace('.', '_')
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            logger.info(f"{stock_code} 处理成功，保存 {len(df)} 条记录")
            return True

        except Exception as e:
            logger.error(f"处理股票 {stock_code} 失败: {e}")
            return None

    def run(self):
        """主运行函数"""
        try:
            print("=" * 60)
            print("Tushare 60分钟股票数据收集程序")
            print("=" * 60)
            print("数据源: Tushare Pro API")
            print("数据频度: 60分钟K线")
            print("数据格式: 包含开高低收成交量成交额及技术指标")
            print("=" * 60)

            self.initialize()

            stock_list = self.get_stock_list()
            logger.info(f"共获取到 {len(stock_list)} 只股票")

            # 测试数据格式
            if stock_list:
                test_stock = stock_list[0]
                logger.info(f"使用 {test_stock} 测试数据格式")
                if not self.test_data_format(test_stock):
                    logger.warning("数据格式测试失败，请检查API权限和网络连接")
                    return

            processed_stocks = self.get_processed_stocks()
            logger.info(f"已处理 {len(processed_stocks)} 只股票")

            remaining_stocks = [code for code in stock_list if code not in processed_stocks]
            logger.info(f"待处理 {len(remaining_stocks)} 只股票")

            if not remaining_stocks:
                logger.info("所有股票都已处理完毕")
                return

            successful = 0
            failed = 0

            print("\n开始处理股票数据...")
            with tqdm(total=len(remaining_stocks), desc="处理进度") as pbar:
                batch_size = 1  # 单个处理，严格控制API调用频率
                for i in range(0, len(remaining_stocks), batch_size):
                    batch = remaining_stocks[i:i + batch_size]

                    for code in batch:
                        try:
                            result = self.process_single_stock(code)
                            if result:
                                successful += 1
                            else:
                                failed += 1
                        except Exception as e:
                            logger.error(f"处理 {code} 时发生错误: {e}")
                            failed += 1
                        finally:
                            pbar.update(1)

                    # 批次间暂停，避免触发API限流（每分钟最多2次）
                    if i + batch_size < len(remaining_stocks):
                        time.sleep(5)  # 增加批次间隔

            print("\n" + "=" * 60)
            logger.info(f"数据获取完成! 成功: {successful}, 失败: {failed}")
            print(f"数据库文件: {self.db_name}")
            print("=" * 60)

        except Exception as e:
            logger.error(f"运行出错: {e}")
        finally:
            self.close()


if __name__ == "__main__":
    print("Tushare 60分钟股票数据收集程序")
    print("基于原 baostock 版本改进，使用 Tushare Pro API")
    print("主要改进：")
    print("1. 使用 Tushare stk_mins 接口获取60分钟数据")
    print("2. 支持更长的历史数据获取")
    print("3. 更稳定的数据质量")
    print("4. 兼容原有的数据结构和技术指标计算")
    print("-" * 60)
    
    collector = StockDataCollector()
    collector.run() 