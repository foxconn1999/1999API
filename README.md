# 1999API
<img width="1620" height="921" alt="圖片1" src="https://github.com/user-attachments/assets/561d362c-21ba-425a-9c1f-41a65550dd42" />

## 模型連結
https://drive.google.com/drive/folders/1sOypQ8iIz3V8hLHxpc83cykdc3toRKut?usp=drive_link

## 專案結構
---

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
    │       └── best.pt         # 需要至雲端連接下載對應.pt檔案並放於./runs/foxconn_file/target底下
    └── src/
        ├── data.py             
        ├── dgcn.py             
        ├── label_graph.py      
        ├── model.py            
        └── utils.py

## 環境需求
- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Pydantic

---

