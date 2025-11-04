# streamlit_app.py
import os, numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import joblib

import re

@st.cache_data(show_spinner=True)
def compute_replacements(df: pd.DataFrame, text_col: str):
    # 常見樣式（順序很重要：先比對 phone / url / email，再數一般數字，避免重複）
    url_pat   = r"(https?://\S+|www\.\S+|\b[a-z0-9.-]+\.(?:com|net|org|co|io|info|biz|edu|gov)(?:/\S*)?)"
    email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    phone_pat = r"(\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b"
    num_pat   = r"\b\d+(?:[\.,]\d+)?\b"

    s = df[text_col].astype(str)

    # 計數
    url_cnt   = s.str.count(url_pat,   flags=re.IGNORECASE).sum()
    email_cnt = s.str.count(email_pat, flags=0).sum()
    phone_cnt = s.str.count(phone_pat, flags=0).sum()

    # 先替換掉較複雜的（避免被 num 抓到）
    cleaned = s.str.replace(phone_pat, "<PHONE>", regex=True)
    cleaned = cleaned.str.replace(url_pat,   "<URL>",   regex=True, flags=re.IGNORECASE)
    cleaned = cleaned.str.replace(email_pat, "<EMAIL>", regex=True)

    # 再數 & 替換一般數字
    num_cnt  = cleaned.str.count(num_pat, flags=0).sum()
    cleaned  = cleaned.str.replace(num_pat, "<NUM>", regex=True)

    counts = pd.Series(
        {"<URL>": int(url_cnt), "<EMAIL>": int(email_cnt), "<PHONE>": int(phone_cnt), "<NUM>": int(num_cnt)}
    )
    return counts, cleaned
def sanitize_text_series(df: pd.DataFrame, text_col: str) -> pd.Series:
    s = df[text_col].astype(str).fillna("").str.strip()
    # 去掉完全空白的列
    s = s[s.str.len() > 0]
    return s


st.set_page_config(page_title="Spam / Ham Classifier — Phase 4 · Visualizations", layout="wide")
st.title("📊 Spam / Ham Classifier — Phase 4 · Visualizations")

# =========================
# Sidebar — Inputs
# =========================
st.sidebar.header("Inputs")
data_path = st.sidebar.text_input("Dataset CSV", value="dataset/sms_spam_no_header.csv")

# 掃一遍欄名，提供下拉選擇
def peek_columns(path):
    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
        if len(cols) == 0:  # 無表頭
            return ["col_0", "col_1"]
        return cols
    except Exception:
        return ["col_0", "col_1"]

cols = peek_columns(data_path)
label_col = st.sidebar.selectbox("Label column", options=cols, index=0)
text_col  = st.sidebar.selectbox("Text column",  options=cols, index=min(1, len(cols)-1))
test_size = st.sidebar.slider("Test size", 0.10, 0.50, 0.20, 0.05)
seed      = st.sidebar.number_input("Seed", value=42, step=1)
threshold = st.sidebar.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)
spam_value_hint = st.sidebar.text_input("Spam label raw value (auto-detect if blank)", value="")
persist   = st.sidebar.checkbox("Persist model to models/", value=False)

# =========================
# Data Loading
# =========================
@st.cache_data(show_spinner=True)
def load_data(path: str, label_col: str, text_col: str):
    # 先試有表頭
    try:
        df = pd.read_csv(path)
        if label_col not in df.columns or text_col not in df.columns:
            # 退回無表頭模式
            df = pd.read_csv(path, names=[label_col, text_col])
    except Exception:
        df = pd.read_csv(path, names=[label_col, text_col], encoding_errors="ignore")
    # 清理型別
    df[text_col] = df[text_col].astype(str).fillna("")
    return df[[label_col, text_col]].copy()

if not os.path.exists(data_path):
    st.error(f"Dataset not found: {data_path}")
    st.stop()

df = load_data(data_path, label_col, text_col)

# =========================
# Label Mapping（彈性）
# =========================
def map_labels(series: pd.Series, spam_value_hint: str):
    s = series.astype(str).str.strip().str.lower()
    # 自動偵測 spam/ham
    if {"spam","ham"}.issubset(set(s.unique())):
        mapped = s.map({"ham":0, "spam":1})
        return mapped.astype(int), {0:"ham",1:"spam"}
    # 如果是 0/1/2... 類數字
    try:
        intlike = series.astype(int)
        # 推測哪個是 spam：若有提示，照提示；否則取「正類=較大值」
        if spam_value_hint.strip() != "":
            pos_raw = int(spam_value_hint)
        else:
            vals = sorted(intlike.unique())
            pos_raw = vals[-1] if len(vals)>1 else vals[0]
        mapped = (intlike == pos_raw).astype(int)
        return mapped, {0:"ham",1:"spam"}
    except Exception:
        pass
    # 若是任意文字標籤（例如 yes/no、pos/neg、sex/spam 之類）
    uniq = s.unique().tolist()
    if spam_value_hint.strip() != "" and spam_value_hint.lower() in uniq:
        pos = spam_value_hint.lower()
    else:
        # 嘗試猜測包含 'spam' 字樣者為正類
        pos = next((u for u in uniq if "spam" in u), uniq[-1])
    mapped = (s == pos).astype(int)
    return mapped, {0:"ham",1:"spam"}

y_mapped, label_names = map_labels(df[label_col], spam_value_hint)
df = pd.DataFrame({label_col: y_mapped, text_col: df[text_col]})

# =========================
# Train (reactive)
# =========================
@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def train_model(df: pd.DataFrame, label_col: str, text_col: str, test_size: float, seed: int):
    # 取出乾淨文字；避免全空
    text_series = sanitize_text_series(df, text_col)
    df2 = pd.DataFrame({label_col: df.loc[text_series.index, label_col],
                        text_col: text_series})
    # class 至少要有兩類
    if df2[label_col].nunique() < 2:
        st.error("Label 只有單一類別，無法訓練模型。請換資料集或修正 Label column。")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        df2[text_col], df2[label_col],
        test_size=test_size, random_state=seed, stratify=df2[label_col]
    )

    # 重點：token_pattern 讓 <URL>/<EMAIL>/<PHONE>/<NUM> 也能被保留
    # 並且先不要用 stop_words，避免把小資料集全過濾掉
    def fit_with_fallback(train_text):
        try:
            vec = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)<[A-Z]+>|\b\w+\b",  # 保留標記與一般詞
                max_features=5000,
                stop_words=None,
            )
            Xtr = vec.fit_transform(train_text)
            return vec, Xtr
        except ValueError:  # empty vocabulary
            # 退而求其次：使用字元 n-gram，比較不怕詞彙表為空
            vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3,5),
                min_df=2, max_features=5000
            )
            Xtr = vec.fit_transform(train_text)
            return vec, Xtr

    vec, Xtr = fit_with_fallback(X_train)
    Xte = vec.transform(X_test)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, y_train)

    return model, vec, (Xtr, Xte, y_train.to_numpy(), y_test.to_numpy())

model, vec, (Xtr, Xte, ytr, yte) = train_model(df, label_col, text_col, test_size, seed)

if persist:
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vec, "models/vectorizer.pkl")

def predict_with_threshold(model, X, thr: float):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:,1]
        pred = (prob >= thr).astype(int)
        return pred, prob
    # fallback：決策分數 → min-max 正規化後套 threshold
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        smin, smax = s.min(), s.max()
        prob = (s - smin) / (smax - smin + 1e-8)
        pred = (prob >= thr).astype(int)
        return pred, prob
    pred = model.predict(X); prob = np.zeros_like(pred, dtype=float)
    return pred, prob

def explain_linear(text: str, vectorizer, model, k=10):
    if not hasattr(model, "coef_"):
        return [("N/A", 0.0)]
    X = vectorizer.transform([text])
    feats = np.array(vectorizer.get_feature_names_out())
    contrib = X.toarray()[0] * model.coef_[0]
    idx = np.argsort(contrib)[::-1][:k]
    return list(zip(feats[idx], contrib[idx].round(4)))

# =========================
# Data Overview
# =========================
st.subheader("Data Overview")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Class distribution**")
    counts = df[label_col].value_counts().rename(label_names)
    st.bar_chart(counts)

with c2:
    st.markdown("**Token replacements in cleaned text (approximate)**")
    counts, cleaned_text = compute_replacements(df, text_col)
    st.bar_chart(counts)

# =========================
# Top Tokens by Class
# =========================
st.subheader("Top Tokens by Class")
def top_tokens_by_class(texts: pd.Series, labels: pd.Series, topn=20):
    # 與訓練一致的規則
    vec_tmp = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)<[A-Z]+>|\b\w+\b",
        max_features=5000,
        stop_words=None,
    )
    if texts.empty:
        return {"ham": pd.DataFrame(columns=["token","count"]),
                "spam": pd.DataFrame(columns=["token","count"])}

    X = vec_tmp.fit_transform(texts)
    vocab = np.array(vec_tmp.get_feature_names_out())
    tops = {}
    for cls in [0, 1]:
        idx_rows = np.where(labels.values == cls)[0]
        if len(idx_rows) == 0:
            tops["ham" if cls==0 else "spam"] = pd.DataFrame(columns=["token","count"])
            continue
        m = X[idx_rows].sum(axis=0).A1
        if m.sum() == 0:
            tops["ham" if cls==0 else "spam"] = pd.DataFrame(columns=["token","count"])
            continue
        top_idx = m.argsort()[::-1][:topn]
        tops["ham" if cls==0 else "spam"] = pd.DataFrame({"token": vocab[top_idx], "count": m[top_idx]})
    return tops

tops = top_tokens_by_class(
    sanitize_text_series(df, text_col),  # 或 cleaned_text（若你想用清洗後）
    df[label_col],
    topn=20
)

tt1, tt2 = st.columns(2)
tt1.markdown("**Class: ham**"); tt1.dataframe(tops["ham"], use_container_width=True, height=320)
tt2.markdown("**Class: spam**"); tt2.dataframe(tops["spam"], use_container_width=True, height=320)

# =========================
# Model Performance (Test)
# =========================
st.subheader("Model Performance (Test)")
y_pred, y_prob = predict_with_threshold(model, Xte, threshold)
cm = confusion_matrix(yte, y_pred)
p, r, f1, _ = precision_recall_fscore_support(yte, y_pred, average="binary", zero_division=0)

m1, m2 = st.columns([2,1])
with m1:
    fig = plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig, clear_figure=True)
with m2:
    st.markdown("**Metrics @ current threshold**")
    st.write(f"Precision: **{p:.3f}**")
    st.write(f"Recall: **{r:.3f}**")
    st.write(f"F1-score: **{f1:.3f}**")

# ---------- Threshold sweep ----------
st.markdown("**Threshold sweep (precision / recall / F1)**")
ts = np.linspace(0.05, 0.95, 19)
prec, rec, f1s = [], [], []
for t in ts:
    pred_t, _ = predict_with_threshold(model, Xte, t)
    P,R,F1,_ = precision_recall_fscore_support(yte, pred_t, average="binary", zero_division=0)
    prec.append(P); rec.append(R); f1s.append(F1)
fig2 = plt.figure()
plt.plot(ts, prec, label="precision"); plt.plot(ts, rec, label="recall"); plt.plot(ts, f1s, label="f1")
plt.xlabel("threshold"); plt.ylabel("score"); plt.ylim(0,1.05); plt.legend()
st.pyplot(fig2, clear_figure=True)

# =========================
# Live Inference
# =========================
st.subheader("Live Inference")
cmsg1, cmsg2 = st.columns([3,2])
with cmsg1:
    text = st.text_area("Enter a message to classify", value="", height=120)
with cmsg2:
    use_examples = st.checkbox("Use spam/ham examples")
    if use_examples:
        ex_spam = "URGENT! You have won a 1,000,000 prize. Call now to claim."
        ex_ham  = "Hey, are we still meeting for lunch tomorrow?"
        ex_choice = st.radio("Pick an example", ["spam","ham"], horizontal=True)
        text = ex_spam if ex_choice=="spam" else ex_ham
    exp_on = st.checkbox("Show token-level explanation", value=True)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a non-empty message.")
    else:
        X = vec.transform([text])
        pred_one, prob_one = predict_with_threshold(model, X, threshold)
        st.write("Prediction:", "🚫 **Spam**" if pred_one[0]==1 else "✅ **Ham**")
        if exp_on:
            rows = explain_linear(text, vec, model, 10)
            st.dataframe({"token":[t for t,_ in rows], "weight":[w for _,w in rows]}, use_container_width=True, height=320)
