import json

# with open("/app/datas/biological_terms.json","r",encoding="utf-8")as file:
#     datas = json.load(file)
#     for data in datas:
#         data["Instruction"]="解釋專有名詞"
#         data["Input"]=data["Proper_noun"]
#         data["Category"]="bio"
#         data["Response"]={}
#         data["Response"]=data["Definition"]
#         del data["Proper_noun"]
#         del data["Definition"]

# with open("/app/datas/biology_terms.jsonl","w",encoding="utf-8")as file:
#     for entry in datas:
#         file.write(json.dumps(entry,ensure_ascii=False)+"\n")

import csv
import openai

from dotenv import load_dotenv
import os

# 載入 .env 檔案
load_dotenv()

# 使用環境變數
api_key = os.getenv('OPENAI_KEY')



client = openai.OpenAI(api_key=api_key)
print(api_key)


def make_jsonl_from_gpt(csv_path,target_jsonl_path,cloume='Input',start_line=0):
    counter = 0
    
    with open(csv_path,newline='')as csvfile:
        reader= csv.DictReader(csvfile)
        with open(target_jsonl_path,"w",encoding="utf-8")as f:
            for row in reader:
                if counter<start_line:
                    continue
                
                try:
                    prompt = f"請用繁體中文提供「{row[cloume]}」的簡短定義，50~100字，需來自維基百科或權威資料。"
                    response = client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                
                    jsonl_entry = {
                        "Instruction": "解釋專有名詞",
                        "Input": row[cloume],
                        "Category": "bio",
                        "Response": answer
                    }

                    f.write(json.dumps(jsonl_entry,ensure_ascii=False)+"\n")
                except Exception as e:
                    print(f"Error processing {row[cloume]}")
                
                print(f"已完成查詢{row[cloume]}")

csv_path="/app/datas/bio_terms_csv.csv"
jsonl_path="/app/datas/biology_terms_from_gpt.jsonl"

# make_jsonl_from_gpt(csv_path,jsonl_path)