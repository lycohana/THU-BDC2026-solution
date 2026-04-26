from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_window_analysis import (  # noqa: E402
    add_branch_diagnostic_features,
    load_raw,
    parse_anchor_args,
    rank_pct,
    realized_returns_for_anchor,
    run_predict_for_anchor,
)
from features import build_history_feature_frame  # noqa: E402
from labels import add_label_o2o_week  # noqa: E402
from portfolio_utils import apply_filter  # noqa: E402
from reranker import (  # noqa: E402
    make_rank_labels,
    predict_rerank_scores,
    select_by_reranker,
    train_lgb_ranker,
    train_top20_classifier,
    train_top5_classifier,
)


BASE_FEATURES = [
    'base_score_pct',
    'score_rank',
    'transformer_rank',
    'lgb_rank',
    'rank_disagreement',
    'ret1_rank',
    'ret5_rank',
    'ret10_rank',
    'ret20_rank',
    'sigma_rank',
    'vol10_rank',
    'amp_rank',
    'amp_mean10_rank',
    'pos20_rank',
    'liq_rank',
    'turnover20_rank',
    'amt_ratio5_rank',
    'to_ratio5_rank',
    'downside_beta60_rank',
    'max_drawdown20_rank',
    'trend_shape_score',
    'crowd_penalty',
]


def _norm_id(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.extract(r"(\d+)")[0].str.zfill(6)


def add_reranker_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_branch_diagnostic_features(df.copy())

    def local_rank(col):
        if col not in out.columns:
            return pd.Series(0.5, index=out.index, dtype=float)
        values = pd.to_numeric(out[col], errors='coerce')
        if '日期' in out.columns:
            return values.groupby(out['日期']).rank(pct=True, method='average').fillna(0.5)
        return rank_pct(values)

    out['score_rank'] = local_rank('score')
    out['base_score_pct'] = out['score_rank']
    out['transformer_rank'] = local_rank('transformer')
    out['lgb_rank'] = local_rank('lgb')
    for rank_col, source_col in {
        'ret10_rank': 'ret10',
        'ret20_rank': 'ret20',
        'vol10_rank': 'vol10',
        'amp_mean10_rank': 'amp_mean10',
        'pos20_rank': 'pos20',
        'turnover20_rank': 'turnover20',
        'amt_ratio5_rank': 'amt_ratio5',
        'to_ratio5_rank': 'to_ratio5',
    }.items():
        out[rank_col] = local_rank(source_col)
    out['crowd_penalty'] = (
        0.65 * ((out['amt_ratio5_rank'] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
        + 0.35 * ((out['to_ratio5_rank'] - 0.80) / 0.20).clip(lower=0.0, upper=1.0)
    )
    out['trend_shape_score'] = (
        0.24 * out['ret10_rank']
        + 0.18 * out['ret20_rank']
        + 0.18 * out['pos20_rank']
        + 0.16 * out['amp_mean10_rank']
        + 0.12 * out['liq_rank']
        + 0.06 * out['turnover20_rank']
        - 0.24 * out['crowd_penalty']
    )
    for col in BASE_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0)
    return out


def build_training_frame(raw_labeled, dates, cutoff_date, history_cache, max_train_dates=100):
    train_dates = [pd.Timestamp(d) for d in dates if pd.Timestamp(d) <= pd.Timestamp(cutoff_date)]
    train_dates = train_dates[-int(max_train_dates):]
    rows = []
    labels = raw_labeled[['股票代码', '日期', 'label_o2o_week']].copy()
    labels['stock_id'] = _norm_id(labels['股票代码'])
    labels['日期'] = pd.to_datetime(labels['日期'])
    for date in train_dates:
        key = date.strftime('%Y-%m-%d')
        if key not in history_cache:
            history_cache[key] = build_history_feature_frame(raw_labeled, asof_date=date)
        feat = history_cache[key].copy()
        day_label = labels[labels['日期'] == date][['stock_id', '日期', 'label_o2o_week']]
        day = feat.merge(day_label, on='stock_id', how='inner')
        if len(day) >= 50:
            rows.append(day)
    if not rows:
        return pd.DataFrame()
    train = pd.concat(rows, ignore_index=True)
    train = make_rank_labels(train, label_col='label_o2o_week', date_col='日期')
    # Historical rows have no model prediction, so model-score ranks are neutral.
    train['score'] = 0.0
    train['transformer'] = train['score']
    train['lgb'] = train['score']
    train = add_reranker_features(train)
    return train.dropna(subset=['label_o2o_week']).copy()


def union_rrf_score(work: pd.DataFrame) -> pd.DataFrame:
    candidates = work.copy()
    source_cols = [
        ('score_lgb_only', 1.0),
        ('transformer', 1.0),
        ('score_balanced', 0.7),
        ('score_defensive_v2', 0.5),
        ('score_conservative_softrisk_v2', 0.5),
        ('score_legal_minrisk', 0.4),
    ]
    candidate_ids = set()
    for col, _ in source_cols:
        if col in candidates.columns:
            topn = 20 if col in {'score_defensive_v2', 'score_legal_minrisk'} else 15
            candidate_ids.update(candidates.nlargest(min(topn, len(candidates)), col)['stock_id'])
    if not candidate_ids:
        candidates['_union_rrf_lcb_score'] = candidates['score']
        return candidates
    candidates = candidates[candidates['stock_id'].isin(candidate_ids)].copy()
    score = pd.Series(0.0, index=candidates.index)
    for col, weight in source_cols:
        if col not in candidates.columns:
            continue
        ranks = candidates[col].rank(ascending=False, method='average')
        score += weight / (30.0 + ranks)
    candidates['_rrf_rank'] = rank_pct(score)
    lgb = rank_pct(candidates['lgb']) if 'lgb' in candidates.columns else 0.5
    risk = candidates.get('tail_risk_score', pd.Series(0.0, index=candidates.index))
    disagreement = candidates.get('rank_disagreement', pd.Series(0.0, index=candidates.index))
    candidates['_union_rrf_lcb_score'] = 0.55 * candidates['_rrf_rank'] + 0.25 * lgb - 0.10 * risk - 0.10 * disagreement
    return candidates


def build_candidate_sets(work: pd.DataFrame):
    current = apply_filter(work.copy(), 'regime_liquidity_anchor_risk_off', liquidity_quantile=0.10, sigma_quantile=0.85)
    reference = work.sort_values('transformer', ascending=False).copy()
    union = union_rrf_score(work).sort_values('_union_rrf_lcb_score', ascending=False)
    trend = apply_filter(work.copy(), 'regime_trend_uncluttered_plus_reversal', liquidity_quantile=0.10, sigma_quantile=0.85)
    return {
        'current_aggressive_top80': current.sort_values('score', ascending=False).head(80),
        'current_aggressive_top120': current.sort_values('score', ascending=False).head(120),
        'current_aggressive_top160': current.sort_values('score', ascending=False).head(160),
        'candidate_union': pd.concat([
            current.sort_values('score', ascending=False).head(120),
            reference.head(30),
            union.head(30),
            trend.sort_values('score', ascending=False).head(30),
        ], ignore_index=True).drop_duplicates('stock_id'),
        'reference_baseline_top30': reference.head(30),
        'union_topn_rrf_lcb_top30': union.head(30),
        'trend_uncluttered_top30': trend.sort_values('score', ascending=False).head(30),
    }


def score_selection(selected: pd.DataFrame, realized: pd.DataFrame):
    detail = selected[['stock_id']].copy()
    detail['stock_id'] = detail['stock_id'].astype(str).str.zfill(6)
    detail = detail.merge(realized[['stock_id', 'realized_ret']], on='stock_id', how='left')
    detail['realized_ret'] = detail['realized_ret'].fillna(0.0)
    return float(detail['realized_ret'].mean()), detail


def rank_map(df: pd.DataFrame, score_col: str):
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    return {sid: i + 1 for i, sid in enumerate(ranked['stock_id'].astype(str))}


def analyze_anchor(raw, raw_labeled, dates, anchor, run_dir, history_cache, args):
    artifacts = run_predict_for_anchor(raw, anchor, run_dir, args.model_dir, not args.no_cache)
    realized, score_window = realized_returns_for_anchor(raw, anchor, label_horizon=args.label_horizon)
    score_df = pd.read_csv(artifacts['score_df_path'], dtype={'stock_id': str})
    score_df['stock_id'] = score_df['stock_id'].astype(str).str.zfill(6)
    work = add_reranker_features(score_df)
    work = work.merge(realized[['stock_id', 'realized_ret']], on='stock_id', how='left')
    work['realized_ret'] = work['realized_ret'].fillna(0.0)

    anchor_idx = dates.index(anchor)
    cutoff_date = dates[anchor_idx - args.label_horizon]
    train_df = build_training_frame(raw_labeled, dates, cutoff_date, history_cache, max_train_dates=args.max_train_dates)
    if len(train_df) < 1000:
        raise RuntimeError(f'not enough reranker train rows for {anchor:%Y-%m-%d}: {len(train_df)}')

    ranker = train_lgb_ranker(train_df, BASE_FEATURES, seed=args.seed)
    top20 = train_top20_classifier(train_df, BASE_FEATURES, seed=args.seed)
    top5 = train_top5_classifier(train_df, BASE_FEATURES, seed=args.seed)
    reranked = predict_rerank_scores(ranker, top20, top5, work, BASE_FEATURES)

    candidate_sets = build_candidate_sets(reranked)
    true_ranked = work.sort_values('realized_ret', ascending=False).reset_index(drop=True)
    true_top20 = set(true_ranked.head(20)['stock_id'])
    true_top5 = set(true_ranked.head(5)['stock_id'])
    old_top120 = candidate_sets['current_aggressive_top120'].copy()
    old_oracle_top120 = float(old_top120.sort_values('realized_ret', ascending=False).head(5)['realized_ret'].mean())

    rows = []
    selected_details = []
    for pool_name in ['current_aggressive_top80', 'current_aggressive_top120', 'current_aggressive_top160', 'candidate_union']:
        pool = candidate_sets[pool_name]
        selected = select_by_reranker(reranked, pool['stock_id'].tolist(), k=5)
        score, detail = score_selection(selected, realized)
        selected_ids = set(detail['stock_id'])
        candidate_recall = len(set(pool['stock_id']) & true_top20) / max(len(true_top20), 1)
        new_gap = old_oracle_top120 - score
        old_gap = old_oracle_top120 - float(old_top120.sort_values('score', ascending=False).head(5)['realized_ret'].mean())
        rows.append({
            'anchor_date': anchor.strftime('%Y-%m-%d'),
            'score_window': score_window,
            'pool': pool_name,
            'selected_score': score,
            'old_ranking_gap_top120': old_gap,
            'new_ranking_gap_top120': new_gap,
            'rerank_gain_ratio': (old_gap - new_gap) / (abs(old_gap) + 1e-12),
            'selected_true_top20_hit_count': len(selected_ids & true_top20),
            'selected_true_top5_hit_count': len(selected_ids & true_top5),
            'candidate_recall_true_top20': candidate_recall,
            'selected_ids': ','.join(detail['stock_id'].tolist()),
            'selected_rets': ','.join(f'{x:.2%}' for x in detail['realized_ret'].tolist()),
        })
        detail['anchor_date'] = anchor.strftime('%Y-%m-%d')
        detail['score_window'] = score_window
        detail['pool'] = pool_name
        selected_details.append(detail)

    ranks_current = rank_map(candidate_sets['current_aggressive_top160'], 'score')
    ranks_ref = rank_map(candidate_sets['reference_baseline_top30'], 'transformer')
    ranks_union = rank_map(candidate_sets['union_topn_rrf_lcb_top30'], '_union_rrf_lcb_score')
    ranks_trend = rank_map(candidate_sets['trend_uncluttered_top30'], 'score')
    ranks_rerank = rank_map(reranked, 'rerank_score')
    ladder = true_ranked.head(20).copy()
    ladder['anchor_date'] = anchor.strftime('%Y-%m-%d')
    ladder['score_window'] = score_window
    ladder['rank_in_current_aggressive'] = ladder['stock_id'].map(ranks_current)
    ladder['rank_in_reference_baseline'] = ladder['stock_id'].map(ranks_ref)
    ladder['rank_in_union'] = ladder['stock_id'].map(ranks_union)
    ladder['rank_in_trend_uncluttered'] = ladder['stock_id'].map(ranks_trend)
    ladder['rank_in_candidate_reranker'] = ladder['stock_id'].map(ranks_rerank)
    feature_cols = ['stock_id', 'realized_ret'] + BASE_FEATURES + ['rerank_score', 'rank_score', 'top20_prob', 'top5_prob']
    ladder = ladder.merge(reranked[feature_cols], on=['stock_id', 'realized_ret'], how='left', suffixes=('', '_feature'))

    return pd.DataFrame(rows), pd.concat(selected_details, ignore_index=True), ladder


def summarize(rows: pd.DataFrame):
    summary = (
        rows.groupby('pool')
        .agg(
            windows=('anchor_date', 'nunique'),
            selected_score_mean=('selected_score', 'mean'),
            selected_score_q10=('selected_score', lambda s: s.quantile(0.10)),
            selected_score_worst=('selected_score', 'min'),
            old_ranking_gap_top120=('old_ranking_gap_top120', 'mean'),
            new_ranking_gap_top120=('new_ranking_gap_top120', 'mean'),
            rerank_gain_ratio=('rerank_gain_ratio', 'mean'),
            selected_true_top20_hit_count=('selected_true_top20_hit_count', 'mean'),
            selected_true_top5_hit_count=('selected_true_top5_hit_count', 'mean'),
            candidate_recall_true_top20=('candidate_recall_true_top20', 'mean'),
        )
        .reset_index()
        .sort_values(['selected_score_mean', 'selected_score_q10'], ascending=[False, False])
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/train_hs300_20260424.csv')
    parser.add_argument('--out-dir', default='temp/reranker_eval')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--anchors', default=None)
    parser.add_argument('--start-anchor', default=None)
    parser.add_argument('--end-anchor', default=None)
    parser.add_argument('--last-n', type=int, default=None)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--min-history-days', type=int, default=100)
    parser.add_argument('--label-horizon', type=int, default=5)
    parser.add_argument('--max-train-dates', type=int, default=90)
    parser.add_argument('--model-dir', default=None)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    raw = load_raw(ROOT / args.raw)
    raw_labeled = add_label_o2o_week(raw, horizon=args.label_horizon)
    dates = list(sorted(raw['日期'].dropna().unique()))
    anchors = parse_anchor_args(args, dates)
    run_dir = ROOT / args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_cache = {}
    rows, details, ladders = [], [], []
    for anchor in anchors:
        anchor = pd.Timestamp(anchor)
        print(f'[reranker] running anchor={anchor:%Y-%m-%d}')
        row, detail, ladder = analyze_anchor(raw, raw_labeled, dates, anchor, run_dir, history_cache, args)
        rows.append(row)
        details.append(detail)
        ladders.append(ladder)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    ladder = pd.concat(ladders, ignore_index=True) if ladders else pd.DataFrame()
    summary = summarize(result) if len(result) else pd.DataFrame()

    result.to_csv(run_dir / 'reranker_window_results.csv', index=False)
    detail.to_csv(run_dir / 'reranker_selected_details.csv', index=False)
    ladder.to_csv(run_dir / 'winner_ladder.csv', index=False)
    summary.to_csv(run_dir / 'reranker_summary.csv', index=False)
    aggregate = {
        'windows': int(result['anchor_date'].nunique()) if len(result) else 0,
        'summary': json.loads(summary.to_json(orient='records')) if len(summary) else [],
        'anchors': [pd.Timestamp(a).strftime('%Y-%m-%d') for a in anchors],
    }
    (run_dir / 'aggregate.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'\nWrote reports to {run_dir}')


if __name__ == '__main__':
    main()
