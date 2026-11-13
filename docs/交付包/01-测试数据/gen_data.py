# -*- coding: utf-8 -*-
"""
EvoResearch 验收测试数据生成器
==============================
三个数据集，分别对应三种「研究难点」：

  A  churn_clean.csv         干净数据   -> 考「不乱改」：没有明显病灶时，能不能不退化
  B  readmission_missing.csv 缺失数据   -> 考「缺失机制」：能不能发现 MNAR 并加指示变量
  C  fraud_imbalanced.csv    不平衡数据 -> 考「指标选择」：能不能不用 accuracy，改看 PR-AUC/F1
                                           （内含一个事后字段 chargeback_filed，是泄漏陷阱）

全部生成过程固定随机种子，任何人重跑都得到逐字节一致的文件。
用法：  python gen_data.py            # 输出到脚本所在目录
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEED = 20261113


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_labels(z, target_rate, rng):
    """二分校准截距 + 固定均匀数采样：正例比例精确可控，且保留 z 的排序信号。"""
    n = len(z)
    u = rng.random(n)
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (u < sigmoid(z + mid)).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (u < sigmoid(z + (lo + hi) / 2)).astype(int)


# ---------------------------------------------------------------- A 干净：客户流失
def make_churn(rng):
    n = 2000
    customer_id = [f"C{100000 + i}" for i in range(n)]
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 72).round(0).astype(int)
    contract = rng.choice(["month-to-month", "one_year", "two_year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.40, 0.45, 0.15])
    payment = rng.choice(
        ["electronic_check", "credit_card", "bank_transfer", "mailed_check"], n,
        p=[0.35, 0.30, 0.20, 0.15])
    monthly = np.clip(rng.normal(70, 22, n), 20, 130).round(2)
    tickets = rng.poisson(1.3, n)
    gb = np.clip(rng.gamma(3.0, 40.0, n), 0, 400).round(1)
    senior = rng.binomial(1, 0.18, n)
    partner = rng.binomial(1, 0.45, n)
    autopay = rng.binomial(1, 0.55, n)
    total = (tenure * monthly * rng.normal(1.0, 0.05, n)).round(2)

    z = (-1.6
         + 1.5 * (contract == "month-to-month")
         + 0.7 * (internet == "fiber")
         - 0.045 * tenure
         + 0.38 * tickets
         + 0.30 * (payment == "electronic_check")
         - 0.55 * autopay
         + 0.35 * senior
         + 0.010 * monthly
         + 0.28 * (internet == "none")
         + 0.7 * (tenure < 6))
    y = binary_labels(z, 0.30, rng)

    df = pd.DataFrame({
        "customer_id": customer_id,          # 陷阱：ID 列，必须丢弃
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "num_support_tickets": tickets,
        "avg_monthly_gb": gb,
        "is_senior": senior,
        "has_partner": partner,
        "autopay_enabled": autopay,
        "churn": y,                          # 目标列
    })
    return df


# ---------------------------------------------------------------- B 缺失：30 天再入院
def make_readmission(rng):
    n = 2400
    age = np.clip(rng.normal(64, 15, n), 18, 95).round(0).astype(int)
    gender = rng.choice(["M", "F"], n, p=[0.48, 0.52])
    num_meds = rng.poisson(9, n) + 1
    prior_adm = rng.poisson(0.7, n)
    los = np.clip(rng.gamma(2.2, 2.6, n), 1, 30).round(0).astype(int)
    smoker = rng.binomial(1, 0.22, n)
    insurance = rng.choice(["medicare", "medicaid", "private", "none"], n,
                           p=[0.5, 0.2, 0.25, 0.05])
    disposition = rng.choice(["home", "home_health", "snf", "rehab"], n,
                             p=[0.55, 0.25, 0.12, 0.08])

    # 潜在严重度：既影响标签，也影响「是否做某项检查」
    severity = rng.normal(0, 1, n)

    bmi = np.clip(rng.normal(29, 6, n) + 0.8 * severity, 15, 55).round(1)
    hba1c = np.clip(rng.normal(6.4, 1.4, n) + 0.55 * severity, 4.5, 14).round(2)
    creat = np.clip(rng.lognormal(0.05, 0.35, n) * (1 + 0.18 * severity), 0.3, 8).round(2)
    hgb = np.clip(rng.normal(13.2, 1.8, n) - 0.5 * severity, 6, 18).round(1)
    sbp = np.clip(rng.normal(132, 18, n) + 3 * severity, 80, 210).round(0).astype(int)

    z = (-2.2
         + 0.030 * age
         + 0.18 * prior_adm
         + 0.10 * los
         + 0.05 * num_meds
         + 0.006 * sbp
         + 0.40 * (disposition == "snf")
         + 0.25 * (disposition == "home_health")
         + 0.28 * smoker
         + 0.09 * (hba1c - 6.4)
         + 0.025 * bmi)
    y = binary_labels(z, 0.28, rng)

    df = pd.DataFrame({
        "patient_id": [f"P{5000 + i}" for i in range(n)],
        "age": age, "gender": gender, "bmi": bmi, "systolic_bp": sbp,
        "hba1c": hba1c, "creatinine": creat, "hemoglobin": hgb,
        "num_medications": num_meds, "prior_admissions": prior_adm,
        "length_of_stay": los, "discharge_disposition": disposition,
        "insurance_type": insurance, "smoker": smoker,
        "readmitted_30d": y,
    })

    # ---- 缺失机制 ----
    # MNAR（与标签相关）：越可能再入院，越可能被安排检查 -> 缺失本身携带信息
    p_hba1c_obs = sigmoid(2.20 - 1.80 * y + 0.15 * severity)
    p_creat_obs = sigmoid(2.60 - 1.40 * y + 0.20 * severity)
    # MCAR（纯随机）
    p_bmi_obs = np.full(n, 0.88)
    p_hgb_obs = np.full(n, 0.85)
    # MAR（与就诊类型相关）：SNF 转出通常带着完整出院记录
    p_disp_obs = np.where(disposition == "snf", 0.98, 0.92)

    df.loc[rng.random(n) > p_hba1c_obs, "hba1c"] = np.nan
    df.loc[rng.random(n) > p_creat_obs, "creatinine"] = np.nan
    df.loc[rng.random(n) > p_bmi_obs, "bmi"] = np.nan
    df.loc[rng.random(n) > p_hgb_obs, "hemoglobin"] = np.nan
    df.loc[rng.random(n) > p_disp_obs, "discharge_disposition"] = np.nan
    return df


# ---------------------------------------------------------------- C 不平衡：信用卡欺诈
def make_fraud(rng):
    n = 8000
    amount = np.clip(rng.lognormal(3.6, 1.0, n), 1, 8000).round(2)
    hour = rng.integers(0, 24, n)
    merchant = rng.choice(
        ["grocery", "restaurant", "online_retail", "travel", "fuel",
         "electronics", "crypto", "gambling"], n,
        p=[0.28, 0.22, 0.20, 0.08, 0.12, 0.06, 0.02, 0.02])
    card_age = np.clip(rng.gamma(2.0, 400, n), 1, 4000).round(0).astype(int)
    txn_24h = rng.poisson(2.2, n)
    avg_7d = np.clip(rng.lognormal(3.4, 0.9, n), 1, 5000).round(2)
    foreign = rng.binomial(1, 0.06, n)
    device_risk = np.clip(rng.beta(2, 5, n), 0, 1).round(3)
    distance = np.clip(rng.lognormal(1.2, 1.1, n), 0, 5000).round(1)
    cvv_retry = rng.choice([0, 1, 2, 3], n, p=[0.90, 0.06, 0.03, 0.01])

    z = (-7.2
         + 3.2 * device_risk
         + 2.4 * foreign
         + 0.9 * (merchant == "crypto")
         + 0.7 * (merchant == "electronics")
         + 0.5 * (merchant == "online_retail")
         + 0.75 * txn_24h
         + 1.0 * (cvv_retry >= 2)
         + 1.4 * ((hour >= 0) & (hour <= 5))
         - 0.0008 * card_age
         + 0.00030 * distance
         + 0.55 * (np.log1p(amount) - 3.6)
         # 交互与非线性：树的模型才会占优，线性模型吃不到
         + 1.8 * device_risk * foreign
         + 1.2 * (txn_24h >= 4) * (np.log1p(amount) > 4.5)
         + 0.9 * ((amount < 15) | (amount > 2000)))
    y = binary_labels(z, 0.04, rng)

    df = pd.DataFrame({
        "txn_id": [f"T{700000 + i}" for i in range(n)],
        "amount": amount,
        "hour_of_day": hour,
        "merchant_category": merchant,
        "card_age_days": card_age,
        "txn_count_24h": txn_24h,
        "avg_amount_7d": avg_7d,
        "is_foreign": foreign,
        "device_risk_score": device_risk,
        "distance_from_home_km": distance,
        "cvv_retry_count": cvv_retry,
        # 陷阱：事后字段。只有欺诈被确认后才会立案，普通争议也会有小概率产生
        "chargeback_filed": np.where(y == 1,
                                     (rng.random(n) < 0.97).astype(int),
                                     (rng.random(n) < 0.005).astype(int)),
        "is_fraud": y,
    })
    return df


TARGETS = ("churn", "readmitted_30d", "is_fraud")


def main():
    rng = np.random.default_rng(SEED)
    jobs = [
        ("churn_clean.csv", make_churn),
        ("readmission_missing.csv", make_readmission),
        ("fraud_imbalanced.csv", make_fraud),
    ]
    for name, fn in jobs:
        df = fn(rng)
        df.to_csv(OUT / name, index=False, encoding="utf-8")
        target = [c for c in df.columns if c in TARGETS][0]
        print(f"{name:26s} rows={len(df):5d} cols={df.shape[1]:2d} "
              f"pos_rate={df[target].mean():.3f} max_missing={df.isna().mean().max():.2f}")


if __name__ == "__main__":
    main()
