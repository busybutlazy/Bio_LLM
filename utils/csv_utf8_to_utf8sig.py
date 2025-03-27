import csv

def utf8_to_utf8sig(source_path,target_path):
    with open(source_path,newline='')as csvfile:
        reader=csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        with open(target_path,'w',encoding="utf-8-sig")as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow(row)



source_path="/app/outputs/evaluate_result/eval_result03270911.csv"
target_path="/app/outputs/evaluate_result/eval_result03270911_v2.csv"
utf8_to_utf8sig(source_path,target_path)