import json
import os
import csv
import openai
from pathlib import Path
from datetime import datetime



def josn2csv(json_path,target_path,overwrite=False,write_header=True):
    if os.path.isfile(target_path) and overwrite==False:
        raise Exception("Sorry, file is already exist. Please choose another path or use 'overwrite==True'.")
    
    


    with open (json_path,'r',encoding="utf-8") as source_f:
        with open (target_path,'w',encoding="utf-8-sig")as target_f:
            try:
                for line in source_f:
                    line_json=json.loads(line)
                    headers=line_json.keys()
                    if write_header:
                        header_row = "".join([key+"," for key in headers])
                        header_row=header_row[:-1]+'\n'
                        target_f.write(header_row)
                        write_header=False
                    row = "".join([keep_n(line_json[header])+"," for header in headers])
                    row = row[:-1]+'\n'
                    target_f.write(row)
            except Exception as e:
                print(f"|Error| {e}")

def keep_n(text):
    text=text.replace('\n','\\n').replace('\\\\\\n','\\n')
    text=text.replace('\n','\\n').replace('\\\\\\n','\\n')
    return text


if __name__=="__main__":
    josn_path=Path("/app/datas/json/gpt_explain_detailed.jsonl")
    csv_path=Path("/app/datas/csv/gpt_explain_detailed.csv")

    josn2csv(josn_path,csv_path,overwrite=True)