from os import listdir,system,makedirs
from os.path import isfile, join,exists

OLDFOLDER_PATH='/workspaces/Bio_LLM/datas/dataset'
TARGETFOLDER_PATH='/workspaces/Bio_LLM/datas/dataset_v2'

onlyfiles = [f for f in listdir(OLDFOLDER_PATH) if isfile(join(OLDFOLDER_PATH,f))]

print(len(onlyfiles))

"""
0~16 104xxxx 104學測題
17~33 105xxxx 105學測題
34~49 106xxxx 106學測題
50~456 001xxxx 講義練習題
457~525 002xxxx 考卷練習題
"""
header={
    0:"104",
    17:"105",
    34:"106",
    50:"001",
    457:"002"
}



head=""
counter=0

sorted_files = sorted(onlyfiles, key=lambda x: int(x.split()[-1].split('.')[0]))

print(sorted_files[-10:])


if not exists(TARGETFOLDER_PATH):
    print("creating the folder...")
    makedirs(TARGETFOLDER_PATH)
else:
    print("folder exist.")

for i in range(len(sorted_files)):
    if i in header:
        head=header[i]
        counter=0

    counter+=1
    newfilename=head+f"{counter:04}"+".png"
    oldfilename=sorted_files[i]
    source_path=join(OLDFOLDER_PATH,oldfilename)
    target_path=join(TARGETFOLDER_PATH,newfilename)
    system(f'cp "{source_path}" "{target_path}"')


    