# 1999API
<img width="1620" height="921" alt="圖片1" src="https://github.com/user-attachments/assets/561d362c-21ba-425a-9c1f-41a65550dd42" />

## 雲端連結
https://drive.google.com/drive/folders/1sOypQ8iIz3V8hLHxpc83cykdc3toRKut?usp=drive_link

- 提供 `/predict` API 進行單筆文本預測
- 自動載入訓練完成的 PyTorch 模型 checkpoint
- 使用 Transformer tokenizer 處理輸入文字
- 載入標籤描述與標籤索引檔案
- 結合 DGCN 與 label-specific representation 進行多標籤分類
- 回傳 Top-K 預測標籤與對應機率分數

## 專案結構

```bash
1999API/
├── README.md
└── runs/
    ├── main.py                 # FastAPI 入口，建立 /predict API
    ├── config.py               # 設定模型路徑、資料路徑與推論參數
    ├── model.py                # API request body 格式定義
    ├── load_model.py           # 載入模型、tokenizer、label 資源
    ├── predict_one.py          # 單筆文字推論流程
    ├── foxconn_file/           # 模型與推論所需資料
    │   ├── label_to_index.json
    │   ├── label_desc_cache_*.json
    │   ├── cache/
    │   │   └── ppmi_eta*.npy
    │   └── target/
    │       └── best.pt
    └── src/
        ├── data.py             # Dataset、collate function、label description tokenize
        ├── dgcn.py             # DGCN 模型模組
        ├── label_graph.py      # PPMI label graph 計算與讀取
        ├── model.py            # MLTCMedoidCLModel 模型架構
        └── utils.py            # JSON 讀取、seed、device 等工具函式
