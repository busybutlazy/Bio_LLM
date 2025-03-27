nowtime=123
data=[{"instruction":"aaa","input":"bbb"},{"instruction":"ccc","input":"ddd"}]

references=["123123123\\n123","123123123\n"]
predictions=["456456456\n","456456456\n"]
with open(f"/app/outputs/evaluate_result/eval_result{nowtime}.csv", "w", encoding="utf-8") as f:
        f.write("instruction,input,reference,prediction,F1\n")
        for i in range(len(data)):
            references[i]=references[i].replace('\n','\\n').replace('\\\\\\n','\\n')
            predictions[i]=predictions[i].replace('\n','\\n').replace('\\\\\\n','\\n')
            row = f'"{data[i]["instruction"]}","{data[i]["input"]}","{references[i]}","{predictions[i]}"\n'
            f.write(row)

print(f"\n📄 已儲存詳細結果至 /app/outputs/evaluate_result/eval_result{nowtime}.csv ✅")