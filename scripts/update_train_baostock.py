import argparse
import atexit
import socket
import time
from datetime import date, timedelta
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]

BAOSTOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)


def log(message: str):
    tqdm.write(message)


def install_socket_timeout(timeout: float):
    socket.setdefaulttimeout(timeout)
    original_socket = socket.socket

    def socket_with_timeout(*args, **kwargs):
        sock = original_socket(*args, **kwargs)
        sock.settimeout(timeout)
        return sock

    socket.socket = socket_with_timeout


def rs_to_df(rs):
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def get_recent_trade_dates(n_days: int, end_date: str | None = None):
    if end_date is None:
        end = date.today()
    else:
        end = pd.to_datetime(end_date).date()

    start = end - timedelta(days=max(500, int(n_days * 2.5)))

    rs = bs.query_trade_dates(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    df = rs_to_df(rs)

    if df.empty:
        raise RuntimeError("没有拿到交易日历，请检查 BaoStock 连接。")

    df = df[df["is_trading_day"] == "1"].copy()
    trade_dates = df["calendar_date"].tolist()

    if len(trade_dates) < n_days:
        raise RuntimeError(f"交易日不足：需要 {n_days} 天，只拿到 {len(trade_dates)} 天")

    return trade_dates[-n_days:]


def get_all_a_stocks(trade_date: str, include_bj: bool = True):
    rs = bs.query_all_stock(day=trade_date)
    df = rs_to_df(rs)

    if df.empty:
        raise RuntimeError(f"{trade_date} 没有拿到股票列表。")

    if include_bj:
        pattern = r"^(sh\.(60|68)|sz\.(000|001|002|003|300|301)|bj\.)"
    else:
        pattern = r"^(sh\.(60|68)|sz\.(000|001|002|003|300|301))"

    df = df[df["code"].str.match(pattern, na=False)].copy()
    return df["code"].drop_duplicates().tolist()


def convert_to_repo_schema(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REPO_COLUMNS)

    df = raw.copy()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "pctChg",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 停牌行对时序窗口和未来收益标签都容易引入噪声。
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == "1"].copy()

    preclose = df["preclose"].replace(0, np.nan)

    out = pd.DataFrame()
    # 仓库 test.csv 使用无交易所前缀的数字代码；predict.py 会再 zfill(6)。
    out["股票代码"] = df["code"].str.split(".").str[-1].astype(int)

    out["日期"] = df["date"]
    out["开盘"] = df["open"]
    out["收盘"] = df["close"]
    out["最高"] = df["high"]
    out["最低"] = df["low"]
    out["成交量"] = df["volume"]
    out["成交额"] = df["amount"]

    # 常见行情口径：振幅 = (最高 - 最低) / 前收盘 * 100。
    out["振幅"] = ((df["high"] - df["low"]) / preclose * 100).replace(
        [np.inf, -np.inf], np.nan
    )

    out["涨跌额"] = df["close"] - df["preclose"]
    out["换手率"] = df["turn"]
    out["涨跌幅"] = df["pctChg"]

    out = out[REPO_COLUMNS]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["股票代码", "日期", "开盘", "收盘", "最高", "最低"])

    num_cols = [c for c in REPO_COLUMNS if c not in ["股票代码", "日期"]]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    return out


def fetch_one_stock(code: str, start_date: str, end_date: str, adjustflag: str):
    rs = bs.query_history_k_data_plus(
        code,
        BAOSTOCK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )

    if rs.error_code != "0":
        log(f"[WARN] {code} failed: {rs.error_msg}")
        return pd.DataFrame()

    return rs_to_df(rs)


def fetch_market(
    start_date: str,
    end_date: str,
    adjustflag: str,
    include_bj: bool,
    sleep: float,
):
    log("[STEP] 获取股票列表")
    codes = get_all_a_stocks(end_date, include_bj=include_bj)
    log(f"[INFO] 股票数量: {len(codes)}")
    log(f"[INFO] 下载区间: {start_date} ~ {end_date}")
    log(f"[INFO] 复权口径 adjustflag={adjustflag}，1=后复权，2=前复权，3=不复权")

    parts = []
    ok_count = 0
    empty_count = 0
    fail_count = 0
    total_rows = 0

    progress = tqdm(
        codes,
        desc="下载个股行情",
        unit="stock",
        dynamic_ncols=True,
        leave=True,
    )
    for code in progress:
        try:
            raw = fetch_one_stock(code, start_date, end_date, adjustflag)
            if not raw.empty:
                parts.append(raw)
                ok_count += 1
                total_rows += len(raw)
            else:
                empty_count += 1
        except Exception as e:
            fail_count += 1
            log(f"[WARN] {code} exception: {e}")

        progress.set_postfix(
            ok=ok_count,
            empty=empty_count,
            fail=fail_count,
            rows=f"{total_rows:,}",
            refresh=False,
        )
        time.sleep(sleep)

    if not parts:
        raise RuntimeError("没有下载到任何行情数据。")

    log("[STEP] 合并并转换为仓库字段")
    raw_all = pd.concat(parts, ignore_index=True)
    return convert_to_repo_schema(raw_all)


def validate_train(df: pd.DataFrame):
    missing = [c for c in REPO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少字段: {missing}")

    dup = df.duplicated(["股票代码", "日期"]).sum()
    if dup:
        raise ValueError(f"存在重复 股票代码+日期: {dup} 行")

    min_date = df["日期"].min()
    max_date = df["日期"].max()
    n_stocks = df["股票代码"].nunique()
    n_rows = len(df)

    log("[CHECK] columns ok")
    log(f"[CHECK] rows={n_rows:,}")
    log(f"[CHECK] stocks={n_stocks:,}")
    log(f"[CHECK] date_range={min_date} ~ {max_date}")
    log("[CHECK] head:")
    log(df.head().to_string())
    log("[CHECK] tail:")
    log(df.tail().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/train.csv")
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--mode", choices=["replace_recent", "append"], default="append")
    parser.add_argument("--overlap-days", type=int, default=90)
    parser.add_argument("--adjustflag", choices=["1", "2", "3"], default="1")
    parser.add_argument("--include-bj", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=0.0,
        help="BaoStock socket 连接和接收超时时间，单位秒；0 表示不改 BaoStock 默认 socket 行为。",
    )
    args = parser.parse_args()

    if args.socket_timeout > 0:
        install_socket_timeout(args.socket_timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lg = None

    with tqdm(total=8, desc="整体进度", unit="step", dynamic_ncols=True) as phase:
        phase.set_postfix_str("登录 BaoStock")
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_code} {lg.error_msg}")
        atexit.register(bs.logout)
        phase.update()

        phase.set_postfix_str("获取交易日历")
        trade_dates = get_recent_trade_dates(args.days, args.end_date)
        recent_start = trade_dates[0]
        end_date = trade_dates[-1]
        phase.update()

        if args.mode == "replace_recent":
            phase.set_postfix_str("replace_recent: 准备下载")
            phase.update()

            phase.set_postfix_str("下载行情")
            new_df = fetch_market(
                start_date=recent_start,
                end_date=end_date,
                adjustflag=args.adjustflag,
                include_bj=args.include_bj,
                sleep=args.sleep,
            )
            phase.update()

            phase.set_postfix_str("整理结果")
            final_df = new_df
            phase.update(2)

        else:
            # append 模式会刷新最近一段重叠区间，覆盖复权变化和偶发漏数。
            phase.set_postfix_str("append: 检查已有 train.csv")
            if out_path.exists() and out_path.stat().st_size > 0:
                old_df = pd.read_csv(out_path)
                old_df = old_df[REPO_COLUMNS].copy()
                old_df["日期"] = pd.to_datetime(old_df["日期"]).dt.strftime("%Y-%m-%d")
                phase.update()

                old_max = old_df["日期"].max()
                phase.set_postfix_str("append: 计算刷新区间")
                all_dates = get_recent_trade_dates(
                    args.days + args.overlap_days + 20, args.end_date
                )

                if old_max in all_dates:
                    idx = all_dates.index(old_max)
                    start_idx = max(0, idx - args.overlap_days + 1)
                    start_date = all_dates[start_idx]
                else:
                    start_date = recent_start

                log(f"[INFO] 已有 train.csv 最新日期: {old_max}")
                log(f"[INFO] append 模式刷新区间: {start_date} ~ {end_date}")
                phase.update()

                phase.set_postfix_str("下载行情")
                new_df = fetch_market(
                    start_date=start_date,
                    end_date=end_date,
                    adjustflag=args.adjustflag,
                    include_bj=args.include_bj,
                    sleep=args.sleep,
                )
                phase.update()

                phase.set_postfix_str("合并去重")
                final_df = pd.concat([old_df, new_df], ignore_index=True)
                final_df = final_df.drop_duplicates(["股票代码", "日期"], keep="last")
                final_df = final_df.sort_values(["股票代码", "日期"]).reset_index(
                    drop=True
                )
                phase.update()

            else:
                log("[INFO] 没有已有 train.csv 或文件为空，改为生成最近 days 个交易日。")
                phase.update(2)

                phase.set_postfix_str("下载行情")
                final_df = fetch_market(
                    start_date=recent_start,
                    end_date=end_date,
                    adjustflag=args.adjustflag,
                    include_bj=args.include_bj,
                    sleep=args.sleep,
                )
                phase.update()

                phase.set_postfix_str("整理结果")
                phase.update()

        phase.set_postfix_str("校验数据")
        final_df = final_df[REPO_COLUMNS]
        validate_train(final_df)
        phase.update()

        phase.set_postfix_str("写入 CSV")
        final_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        phase.update()
        phase.set_postfix_str("完成")
        log(f"[DONE] 已写入: {out_path}")

    if lg is not None:
        bs.logout()
        atexit.unregister(bs.logout)


if __name__ == "__main__":
    main()
