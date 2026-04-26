import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker


def _clean_X(df, feature_cols):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    return X.fillna(0.0).astype(np.float32)


def _zscore(values):
    x = np.asarray(values, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def make_rank_labels(df, label_col='label_o2o_week', date_col='日期'):
    out = df.copy()
    out['future_rank'] = out.groupby(date_col)[label_col].rank(method='first', ascending=False)
    out['rank_label'] = 0
    out.loc[out['future_rank'] <= 120, 'rank_label'] = 1
    out.loc[out['future_rank'] <= 60, 'rank_label'] = 2
    out.loc[out['future_rank'] <= 20, 'rank_label'] = 3
    out.loc[out['future_rank'] <= 5, 'rank_label'] = 4
    out['is_top20'] = (out['future_rank'] <= 20).astype(int)
    out['is_top5'] = (out['future_rank'] <= 5).astype(int)
    return out


def train_lgb_ranker(train_df, feature_cols, date_col='日期', label_col='rank_label', seed=42):
    data = train_df.sort_values([date_col, 'stock_id']).reset_index(drop=True)
    group = data.groupby(date_col, sort=False).size().to_numpy()
    model = LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[5, 20],
        learning_rate=0.04,
        n_estimators=220,
        num_leaves=31,
        min_child_samples=48,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(data, feature_cols), data[label_col].astype(int), group=group)
    return model


def train_top20_classifier(train_df, feature_cols, label_col='is_top20', seed=42):
    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.035,
        n_estimators=180,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(train_df, feature_cols), train_df[label_col].astype(int))
    return model


def train_top5_classifier(train_df, feature_cols, label_col='is_top5', seed=42):
    model = LGBMClassifier(
        objective='binary',
        learning_rate=0.03,
        n_estimators=180,
        num_leaves=15,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(_clean_X(train_df, feature_cols), train_df[label_col].astype(int))
    return model


def _positive_prob(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def predict_rerank_scores(ranker, top20_clf, top5_clf, pred_df, feature_cols):
    out = pred_df.copy()
    X = _clean_X(out, feature_cols)
    out['rank_score'] = np.asarray(ranker.predict(X), dtype=np.float64)
    out['top20_prob'] = _positive_prob(top20_clf, X)
    out['top5_prob'] = _positive_prob(top5_clf, X)
    out['rerank_score'] = (
        0.50 * _zscore(out['rank_score'])
        + 0.40 * _zscore(out['top20_prob'])
        + 0.10 * _zscore(out['top5_prob'])
    )
    return out


def select_by_reranker(pred_df, candidate_ids=None, k=5):
    out = pred_df.copy()
    if candidate_ids is not None:
        ids = set(str(x).zfill(6) for x in candidate_ids)
        out = out[out['stock_id'].astype(str).str.zfill(6).isin(ids)].copy()
    if len(out) < k:
        out = pred_df.copy()
    return out.sort_values('rerank_score', ascending=False).head(k).copy()
