import os 
import shutil
import random

dataset_path="/app/datas/dataset"
label_path="/app/datas/dataset"

output_path="/app/datas/dataset_split_all"

TRAIN_IMG_DIR=os.path.join(output_path,"images/train")

VAL_IMG_DIR=os.path.join(output_path,"images/val")

TRAIN_LABEL_DIR=os.path.join(output_path,"labels/train")

VAL_LABEL_DIR=os.path.join(output_path,"labels/val")

# 從資料集路徑獲得檔名，並分割資料集
def split_tain_val(dataset_path,split_ratio=0.8)->tuple[list[str],list[str]]:
    """
    This function searches for image files with extensions ".jpg", ".png", and ".jpeg" in the specified directory.
    It then shuffles the filenames randomly and splits them into two lists based on the given split ratio.
    The function returns two lists of filenames: one for training and one for validation.
    """
    image_files = [f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".png", ".jpeg"))]

    random.shuffle(image_files)

    split_idx = int (len(image_files)*split_ratio)

    train_files=image_files[:split_idx]
    val_files=image_files[split_idx:]
    return train_files,val_files

def copy_files(files,src_img_folder,src_label_folder,dataset_img_folder,dataset_label_folder):
    for file in files:
        src_img=os.path.join(src_img_folder,file)
        dataset_img = os.path.join(dataset_img_folder,file)
        
        label_file=os.path.splitext(file)[0]+".txt"
        src_label =os.path.join(src_label_folder,label_file)
        dataset_label=os.path.join(dataset_label_folder,label_file)

        shutil.copyfile(src_img,dataset_img)
        if os.path.exists(src_label):
            shutil.copyfile(src_label,dataset_label)
        else:
            print("can't found label file.")

# 第一次分割dataset用

# for folder in [TRAIN_IMG_DIR,VAL_IMG_DIR,TRAIN_LABEL_DIR,VAL_LABEL_DIR]:
#         if os.path.isdir(folder):
#             print("dir exist.")
#         else:
#             os.mkdir(folder)
#             print(f"building dir {folder}.")

# train_files,val_files=split_tain_val(dataset_path,0.8)

# copy_files(train_files,dataset_path,dataset_path,TRAIN_IMG_DIR,TRAIN_LABEL_DIR)
# copy_files(val_files,dataset_path,dataset_path,VAL_IMG_DIR,VAL_LABEL_DIR)

# print(f"✅ 分割完成！訓練集: {len(train_files)}，驗證集: {len(val_files)}")


from ultralytics import YOLO

model = YOLO("/app/yolo11s.pt")

model.train(data="/app/datas/dataset.yml",epochs=1000,batch=16,imgsz=640,workers=2,device="cuda")

model.export(format="onnx")  # 轉換為 ONNX 格式