import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd
import tinyshare as ts

# --- 配置 ---
# 从环境变量中获取 Tushare Token
TUSHARE_TOKEN = "gOq6B37pCOWApMc1Ha28Ff0i8WBaQyEvVv8pIDQ0AER2r09UV2kcThwgfdf9a8a9"
if not TUSHARE_TOKEN:
    raise ValueError("请设置 TUSHARE_TOKEN 环境变量")

# 数据存储路径

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, '..', 'tushare_data')
DAILY_DATA_PATH = os.path.join(DATA_PATH, "daily")
STOCK_BASIC_FILE = os.path.join(DATA_PATH, "all_stocks.csv")
SYNC_STATE_FILE = os.path.join(DATA_PATH, "daily_sync_state.json")
BATCH_FLUSH_THRESHOLD = 20000  # 批量写入阈值，避免频繁 IO

# 创建数据存储目录
os.makedirs(DAILY_DATA_PATH, exist_ok=True)

# 初始化 Tushare Pro API
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def _load_sync_state():
    if not os.path.exists(SYNC_STATE_FILE):
        return {}
    try:
        with open(SYNC_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
            if isinstance(state, dict):
                return state
            print("同步状态文件格式异常，忽略并重新开始。")
    except Exception as err:
        print(f"读取同步状态文件失败，忽略并重新开始: {err}")
    return {}


def _save_sync_state(**kwargs):
    state = _load_sync_state()
    state.update({k: v for k, v in kwargs.items() if v is not None})
    with open(SYNC_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _load_reference_date():
    state = _load_sync_state()
    return state.get("reference_trade_date")


def _save_reference_date(reference_date: str):
    _save_sync_state(reference_trade_date=reference_date)


def _load_last_synced_date():
    state = _load_sync_state()
    return state.get("last_trade_date")


def _save_last_synced_date(date_str: str):
    _save_sync_state(last_trade_date=date_str)


def _get_trade_dates(start_date: str, end_date: str):
    """
    使用交易日历计算在指定时间范围内的所有开盘日
    """
    cal = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields="cal_date,is_open",
    )
    trade_dates = cal.loc[cal["is_open"] == 1, "cal_date"].tolist()
    trade_dates.sort()
    return trade_dates


def _fetch_reference_adj_factors(reference_date: str, ts_codes):
    """
    获取参考日期的复权因子，作为所有股票的前复权基准
    """
    try:
        ref_df = pro.adj_factor(
            trade_date=reference_date,
            fields="ts_code,trade_date,adj_factor",
        )
    except Exception as err:
        print(f"获取参考复权因子失败: {err}")
        ref_df = None

    if ref_df is None or ref_df.empty:
        print(f"参考日 {reference_date} 未返回复权因子，默认所有股票为 1")
        ref_map = {code: 1.0 for code in ts_codes}
    else:
        required_cols = {"ts_code", "adj_factor"}
        if not required_cols.issubset(set(ref_df.columns)):
            missing = required_cols.difference(set(ref_df.columns))
            print(
                f"参考日 {reference_date} 缺少列 {missing}，默认所有股票复权因子为 1"
            )
            ref_map = {code: 1.0 for code in ts_codes}
        else:
            ref_map = dict(zip(ref_df["ts_code"], ref_df["adj_factor"]))

    missing_codes = [code for code in ts_codes if code not in ref_map]
    if missing_codes:
        print(f"参考日 {reference_date} 缺少 {len(missing_codes)} 只股票的复权因子，默认使用 1")
        for code in missing_codes:
            ref_map[code] = 1.0
    return ref_map


def _increment_date(date_str: str):
    dt = datetime.strptime(date_str, "%Y%m%d")
    return (dt + timedelta(days=1)).strftime("%Y%m%d")


def _read_last_data_row(file_path):
    """
    高效读取 CSV 的最后一行非空数据
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return None
            buffer = bytearray()
            pointer = f.tell() - 1
            while pointer >= 0:
                f.seek(pointer)
                char = f.read(1)
                if char == b"\n":
                    if buffer:
                        break
                else:
                    buffer.extend(char)
                pointer -= 1
            if not buffer:
                return None
            line = buffer[::-1].decode("utf-8").strip()
            return line if line and not line.startswith("Date") else None
    except Exception as err:
        print(f"读取 {file_path} 最后一行失败: {err}")
        return None


def _infer_last_synced_from_existing_files(ts_codes):
    """
    如果没有状态文件，则根据已有 CSV 推断最新同步日期
    """
    date_counter = Counter()
    latest_date = None
    for ts_code in ts_codes:
        file_path = os.path.join(DAILY_DATA_PATH, f"{ts_code}.csv")
        if not os.path.exists(file_path):
            continue
        last_row = _read_last_data_row(file_path)
        if not last_row:
            continue
        date_part = last_row.split(",", 1)[0].strip()
        try:
            dt = datetime.strptime(date_part, "%Y-%m-%d")
        except ValueError:
            continue
        date_str = dt.strftime("%Y%m%d")
        date_counter[date_str] += 1
        if latest_date is None or date_str > latest_date:
            latest_date = date_str

    if not latest_date:
        return None

    dominant_date, dominant_count = date_counter.most_common(1)[0]
    coverage = dominant_count / max(len(ts_codes), 1)
    # 当大部分股票都停留在同一日期时，优先相信该日期；否则使用全局最大日期
    inferred = dominant_date if coverage >= 0.6 else latest_date
    print(
        f"根据已有 CSV 推断最新交易日为 {inferred} "
        f"(覆盖率 {coverage:.0%})，将从下一交易日开始增量。"
    )
    return inferred


def _flush_pending_batches(pending_batches):
    """
    将缓存中的多只股票数据写入各自的 CSV
    """
    for ts_code, frames in pending_batches.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        df.index.name = "Date"
        file_path = os.path.join(DAILY_DATA_PATH, f"{ts_code}.csv")
        file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        df.to_csv(file_path, mode="a", header=not file_exists)
    pending_batches.clear()


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
    按交易日批量同步所有 A 股的后复权日线数据。
    通过“按天取全市场”方式，将 API 调用次数从“按股票取数据”显著降低，
    极大提升增量更新时的吞吐效率。
    """
    if not os.path.exists(STOCK_BASIC_FILE):
        print(f"未找到股票基本信息文件: {STOCK_BASIC_FILE}")
        print("请先运行 sync_all_stock_basic() 函数同步基本信息。")
        stock_basic = sync_all_stock_basic()
        if stock_basic is None:
            return
    else:
        stock_basic = pd.read_csv(STOCK_BASIC_FILE)

    if stock_basic.empty:
        print("股票基础信息为空，终止同步。")
        return

    ts_codes = stock_basic["ts_code"].astype(str).tolist()
    ts_code_set = set(ts_codes)

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    last_synced_date = _load_last_synced_date()
    inferred_from_files = None
    effective_start = start_date
    if last_synced_date:
        next_day = _increment_date(last_synced_date)
        if next_day > effective_start:
            effective_start = next_day
    else:
        inferred_from_files = _infer_last_synced_from_existing_files(ts_codes)
        if inferred_from_files:
            inferred_next = _increment_date(inferred_from_files)
            if inferred_next > effective_start:
                effective_start = inferred_next

    if effective_start > end_date:
        print("本地数据已是最新，无需同步。")
        return

    trade_dates = _get_trade_dates(effective_start, end_date)
    if not trade_dates:
        print("指定区间内没有可交易的日期。")
        return

    reference_date = _load_reference_date()
    if reference_date:
        print(f"沿用历史基准 {reference_date} 作为前复权基准。")
    else:
        reference_date = trade_dates[-1]
        _save_reference_date(reference_date)
        print(f"首次运行，使用 {reference_date} 的复权因子作为前复权基准。")
    reference_adj_factors = _fetch_reference_adj_factors(reference_date, ts_code_set)

    pending_batches = defaultdict(list)
    pending_rows = 0
    total_days = len(trade_dates)
    latest_processed = None

    for idx, trade_date in enumerate(trade_dates, start=1):
        print(f"[{idx}/{total_days}] 同步交易日 {trade_date} ...")

        try:
            daily_df = pro.daily(
                trade_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,vol",
            )
        except Exception as err:
            print(f"  -> 获取 {trade_date} 行情失败: {err}")
            continue

        if daily_df is None or daily_df.empty:
            print("  -> 当日无交易数据，跳过。")
            continue

        daily_df = daily_df[daily_df["ts_code"].isin(ts_code_set)]
        if daily_df.empty:
            continue

        try:
            adj_df = pro.adj_factor(
                trade_date=trade_date,
                fields="ts_code,trade_date,adj_factor",
            )
        except Exception as err:
            print(f"  -> 获取 {trade_date} 复权因子失败: {err}，使用 1 作为默认值。")
            adj_df = pd.DataFrame(
                {
                    "ts_code": daily_df["ts_code"],
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            )

        if adj_df is None or adj_df.empty:
            adj_df = pd.DataFrame(
                {
                    "ts_code": daily_df["ts_code"],
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            )

        merged = pd.merge(
            daily_df,
            adj_df,
            on=["ts_code", "trade_date"],
            how="left",
        )

        merged["adj_factor"] = merged["adj_factor"].fillna(
            merged["ts_code"].map(reference_adj_factors)
        )
        merged["ref_factor"] = merged["ts_code"].map(reference_adj_factors)
        merged["ref_factor"] = merged["ref_factor"].fillna(merged["adj_factor"])
        merged.loc[merged["ref_factor"] == 0, "ref_factor"] = 1.0

        merged["adj_ratio"] = merged["adj_factor"] / merged["ref_factor"]
        merged["adj_ratio"] = merged["adj_ratio"].replace(
            [float("inf"), -float("inf")], 1.0
        )
        merged["adj_ratio"] = merged["adj_ratio"].fillna(1.0)

        for price_col in ("open", "high", "low", "close"):
            merged[price_col] = merged[price_col] * merged["adj_ratio"]

        ratio_for_vol = merged["adj_ratio"].replace(0, 1.0)
        merged["vol"] = merged["vol"] / ratio_for_vol

        merged.rename(
            columns={
                "trade_date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "vol": "Volume",
            },
            inplace=True,
        )
        merged["Date"] = pd.to_datetime(merged["Date"], format="%Y%m%d")
        merged = merged[["ts_code", "Date", "Open", "High", "Low", "Close", "Volume"]]

        grouped = merged.groupby("ts_code")
        for ts_code, group in grouped:
            batch = group.drop(columns=["ts_code"])
            pending_batches[ts_code].append(batch)
            pending_rows += len(batch)

        if pending_rows >= BATCH_FLUSH_THRESHOLD:
            _flush_pending_batches(pending_batches)
            pending_rows = 0

        latest_processed = trade_date

    if pending_rows:
        _flush_pending_batches(pending_batches)

    if latest_processed:
        _save_last_synced_date(latest_processed)
        print(f"完成 {effective_start} 至 {latest_processed} 的同步。")
    else:
        if inferred_from_files:
            _save_last_synced_date(inferred_from_files)
            print(f"未拉取到新增交易日，但已记录最新本地日期 {inferred_from_files}。")
        print("未成功同步任何交易日。")


if __name__ == "__main__":
    print("开始执行数据同步任务...")
    
    # 步骤1: 同步股票基本信息
    sync_all_stock_basic()
    
    # 步骤2: 同步所有股票的日线数据
    # 您可以根据需要修改这里的起止日期
    sync_all_daily_data(start_date="20200101")
    
    # print("所有数据同步任务完成！")
