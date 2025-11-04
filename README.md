# 📧 垃圾郵件分類器（Spam / Ham Classifier）— 第四階段：視覺化分析

本專案實作一個完整的 **垃圾郵件分類系統（Spam Email Classifier）**，
使用 **機器學習 (Machine Learning)** 與 **Streamlit** 建立互動式儀表板。
本專案資料與前處理流程參考自 **Packt 出版《Hands-On Artificial Intelligence for Cybersecurity》 第三章**，
並延伸設計了更豐富的視覺化模組與可解釋 AI（Explainable AI）介面，能顯示詞級權重貢獻。

---

## 🚀 功能特色

* **側欄資料與參數切換**：可即時更換資料集（CSV）與參數（`test size`、`seed`、`threshold`）。
* **自動化訓練流程**：TF-IDF + Logistic Regression 模型，資料或參數變動即自動重新訓練。
* **資料概覽（Data Overview）**

  * 分類比例圖（Class Distribution）
  * 特殊標記替換統計（`<URL>`、`<EMAIL>`、`<PHONE>`、`<NUM>`）
* **類別關鍵詞分析（Top Tokens by Class）**
  顯示每個類別最具代表性的關鍵詞。
* **模型效能（Model Performance）**

  * 混淆矩陣（Confusion Matrix）
  * 精確率 / 召回率 / F1 指標
  * 門檻掃描曲線（Threshold Sweep Curve）
* **即時推論與解釋（Live Inference + Explainability）**

  * 即時輸入郵件內容 → 立即顯示分類結果（`✅ Ham` / `🚫 Spam`）
  * 可選擇顯示詞級權重表格（Token-level contributions）

---

## 🧩 專案架構

```
2025ML-spamEmail/
├─ dataset/
│  ├─ sms_spam_no_header.csv
│  ├─ sms_spam_perceptron.csv
│  └─ phishing_dataset.csv
├─ models/                 # 可選擇儲存模型檔
├─ streamlit_app.py        # 主程式（Streamlit 介面）
├─ requirements.txt        # 套件依賴
├─ README.md
└─ openspec/               # (選用) OpenSpec 工作流程檔案
```

---

## 🧰 技術架構

| 類別       | 工具 / 函式庫                          |
| -------- | --------------------------------- |
| 語言       | Python 3.8+                       |
| 前端框架     | [Streamlit](https://streamlit.io) |
| 機器學習與前處理 | `scikit-learn`, `pandas`, `numpy` |
| 視覺化      | `matplotlib`, Streamlit 內建圖表      |
| 可解釋性     | Logistic Regression 權重詞級分析        |

---

## ⚙️ 安裝與執行

1. **下載專案**

   ```bash
   git clone https://github.com/huanchen1107/2025ML-spamEmail.git
   cd 2025ML-spamEmail
   ```

2. **安裝依賴套件**

   ```bash
   pip install -r requirements.txt
   ```

3. **執行應用**

   ```bash
   streamlit run streamlit_app.py
   ```

4. 開啟瀏覽器進入：
   👉 `http://localhost:8501`

---

## 🧠 模型邏輯說明

* 輸入文字 → 經 TF-IDF 向量化（`token_pattern = (?u)<[A-Z]+>|\b\w+\b`）
* 使用 Logistic Regression（最大迭代 1000）訓練
* 輸出 `p(spam)` 機率 → 以 threshold（預設 0.5）決策
* 可解釋性：每個詞權重 = 詞 TF-IDF 值 × 模型係數

---

## 📊 視覺化區塊說明

| 區塊                      | 說明                              |
| ----------------------- | ------------------------------- |
| **Data Overview**       | 類別分布 + 特殊標記統計                   |
| **Top Tokens by Class** | 各類別最常見前 20 個關鍵詞                 |
| **Model Performance**   | 混淆矩陣與評估指標                       |
| **Threshold Sweep**     | Precision / Recall / F1 與門檻值關係圖 |
| **Live Inference**      | 可互動輸入文字並顯示詞權重貢獻                 |

---

## 🧪 範例輸出

```
Prediction: ✅ Ham
token      weight
------     -------
hello      0.0856
help       0.0000
urgent     0.0000
...
```

---

## 📚 資料集來源

原始資料庫：
[PacktPublishing / Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)

使用資料集：

* `sms_spam_no_header.csv`
* `sms_spam_perceptron.csv`
* `phishing_dataset.csv`
