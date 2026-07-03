# 1999API
<img width="1620" height="921" alt="圖片1" src="https://github.com/user-attachments/assets/561d362c-21ba-425a-9c1f-41a65550dd42" />

## 專案結構
---

```bash
1999API/
├── README.md
└── runs/
    ├── main.py                     # FastAPI 入口，建立 /predict, /particular_predict API
    ├── config.py                   # 設定模型路徑、資料路徑與推論參數
    ├── model.py                    # API request body 與 model_manager cache定義
    ├── load_model.py               # 載入模型、tokenizer、label 資源
    ├── predict_one.py              # 單筆文字推論流程
    ├── Agenda/                     # 局處室分類 標籤敘述&模型目錄
    ├── level1/                     # 第一階層(類別分類) 標籤敘述&模型目錄
    ├── level2/                     # 第二階層(主項分類) 標籤敘述&模型目錄
    ├── level3/                     # 第三階層(子項分類) 標籤敘述&模型目錄, 附註: Agenda, level1~3資料夾需到雲端下載
    └── src/
        ├── data.py             
        ├── dgcn.py             
        ├── label_graph.py      
        ├── model.py            
        └── utils.py
```

---

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

## 安裝套件
```bash
pip install fastapi uvicorn torch transformers numpy pandas pydantic
```

## 輸入格式
### Request Body

```json
/predict
{
  "Text": "請輸入要進行分類預測的文章內容",
  "K": 5
}
```
```json
/particular_predict
{
  "Text": "請輸入要進行分類預測的文章內容",
  "K": 5
  "Category": "查詢類別"
}
```

## 輸出格式
### Response Body

```json
/predict
{
    "agenda":
    {
        "局處室1": 0.9866,
        "局處室2": 0.5489,
    }
    "level1":
    {
        "類別1": 0.9665,
        "類別2": 0.0016,
    },
    "level2":
    {
        "類別1":
        {
            "主項1": 0.8567,
            "主項2": 0.5645
        },
        "類別2":
        {
            ...
        }
    },
    "level3":
    {
        "類別1":
        {
            "主項1":
            {
                "子項1": 0.9618
                "子項2": 0.1234
            },
            "主項2":
            {
                "子項1": 0.1248
                "子項2": 0.0012
            }
        },
        "類別2":
        {
            ...
        }
    }
}
```
```json
/particular_predict
{
    "level1": "輸入類別名稱",
    "level2":
    {
        "輸入類別名稱":
        {
            "主項1": 0.8567,
            "主項2": 0.5645
        }
    },
    "level3":
    {
        "輸入類別名稱":
        {
            "主項1":
            {
                "子項1": 0.9618
                "子項2": 0.1234
            },
            "主項2":
            {
                "子項1": 0.1248
                "子項2": 0.0012
            }
        }
    }
}
```

## 呼叫範例
### 使用 cURL 呼叫

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"Text\": \"輸入文章內容。\", \"K\": 5}"
```

### 使用 Python 呼叫

```python
import requests

url = "http://127.0.0.1:8000/predict" # /predict ===> API of POST

payload = {
    "Text": "輸入文章內容。",
    "K": 5
}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```
