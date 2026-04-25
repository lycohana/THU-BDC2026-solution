import argparse
import atexit
import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from update_train_baostock import (  # noqa: E402
    REPO_COLUMNS,
    convert_to_repo_schema,
    fetch_one_stock,
    install_socket_timeout,
    rs_to_df,
    validate_train,
)


def load_hs300_codes(list_path: Path | None) -> pd.DataFrame:
    if list_path is not None:
        df = pd.read_csv(list_path)
        if "code" not in df.columns:
            raise ValueError(f"沪深300列表缺少 code 列: {list_path}")
        return df

    rs = bs.query_hs300_stocks()
    if rs.error_code != "0":
        raise RuntimeError(f"获取沪深300列表失败: {rs.error_code} {rs.error_msg}")
    df = rs_to_df(rs)
    if df.empty:
        raise RuntimeError("沪深300列表为空")
    return df


def fetch_hs300_history(
    codes: list[str],
    start_date: str,
    end_date: str,
    adjustflag: str,
    sleep: float,
) -> pd.DataFrame:
    parts = []
    ok_count = 0
    empty_count = 0
    fail_count = 0
    total_rows = 0

    progress = tqdm(codes, desc="下载沪深300行情", unit="stock", dynamic_ncols=True)
    for code in progress:
        try:
            raw = fetch_one_stock(code, start_date, end_date, adjustflag)
            if raw.empty:
                empty_count += 1
            else:
                parts.append(raw)
                ok_count += 1
                total_rows += len(raw)
        except Exception as exc:
            fail_count += 1
            tqdm.write(f"[WARN] {code} failed: {exc}")

        progress.set_postfix(
            ok=ok_count,
            empty=empty_count,
            fail=fail_count,
            rows=f"{total_rows:,}",
            refresh=False,
        )
        time.sleep(sleep)

    if not parts:
        raise RuntimeError("没有下载到任何沪深300行情数据")

    raw_all = pd.concat(parts, ignore_index=True)
    return convert_to_repo_schema(raw_all)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建沪深300专用训练数据集，输出为仓库 train.csv 相同字段。"
    )
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default="2026-04-24")
    parser.add_argument("--out", default="data/train_hs300_20260424.csv")
    parser.add_argument("--hs300-list", default=None, help="可选：已有沪深300列表 CSV")
    parser.add_argument("--list-out", default="data/hs300_stock_list_current.csv")
    parser.add_argument("--adjustflag", choices=["1", "2", "3"], default="1")
    parser.add_argument("--sleep", type=float, default=0.08)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.socket_timeout > 0:
        install_socket_timeout(args.socket_timeout)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_out = Path(args.list_out) if args.list_out else None
    if list_out is not None:
        list_out.parent.mkdir(parents=True, exist_ok=True)

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_code} {lg.error_msg}")
    atexit.register(bs.logout)

    try:
        hs300_df = load_hs300_codes(Path(args.hs300_list) if args.hs300_list else None)
        hs300_df = hs300_df.drop_duplicates("code").sort_values("code").reset_index(
            drop=True
        )
        codes = hs300_df["code"].astype(str).tolist()

        if list_out is not None:
            hs300_df.to_csv(list_out, index=False, encoding="utf-8-sig")
            tqdm.write(f"[INFO] 沪深300列表已写入: {list_out}")

        tqdm.write(f"[INFO] 沪深300股票数: {len(codes)}")
        tqdm.write(f"[INFO] 下载区间: {args.start_date} ~ {args.end_date}")
        tqdm.write(
            f"[INFO] 复权口径 adjustflag={args.adjustflag}，1=后复权，2=前复权，3=不复权"
        )

        df = fetch_hs300_history(
            codes=codes,
            start_date=args.start_date,
            end_date=args.end_date,
            adjustflag=args.adjustflag,
            sleep=args.sleep,
        )
        df = df[REPO_COLUMNS].sort_values(["股票代码", "日期"]).reset_index(drop=True)

        validate_train(df)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"[DONE] 已写入: {out_path}")
    finally:
        bs.logout()
        atexit.unregister(bs.logout)


if __name__ == "__main__":
    main()
