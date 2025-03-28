import csv
import json

from pathlib import Path

def csv_to_json(csv_path,target_path,mode="a"):
    counter=0
    try:
        with open(csv_path,newline='')as csv_file:
            reader=csv.DictReader(csv_file)
            dict_header = reader.fieldnames
            print(f"dict_header:{dict_header}")
            with open(target_path,mode,encoding="utf-8-sig")as target_f:
                
                for row in reader:
                    
                    jsonl_entry = {
                        "prompt": f"請{row[dict_header[0]]}:{row[dict_header[1]]}",
                        "chosen": row[dict_header[2]],
                        "rejected": row[dict_header[3]],
                    }
                    counter+=1
                    target_f.write(json.dumps(jsonl_entry,ensure_ascii=False)+"\n")
    
    except Exception as e:
        print(f"Error: {e}")

    print(f"輸出資料完畢。共{counter}筆資料。")



if __name__ == "__main__":
    csv_paths=["/app/outputs/evaluate_result/eval_result03272130.csv","/app/outputs/evaluate_result/eval_result03280634.csv"]
    target_path=Path("/app/datas/json/dpo_train_1.jsonl")

    for csv_path in csv_paths:
        csv_to_json(Path(csv_path),target_path)
