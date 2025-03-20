from os import listdir,system,makedirs
from os.path import isfile, join,exists

OLDFOLDER_PATH='/app/datas/raw_data/2_16add_screen_shot'
TARGETFOLDER_PATH='/app/datas/dataset_add_02_16'

onlyfiles = [f for f in listdir(OLDFOLDER_PATH) if isfile(join(OLDFOLDER_PATH,f))]

print(len(onlyfiles))

"""
0~18 108xxxx 108學測題
19~33 107xxxx 107學測題
34~52 109xxxx 109學測題
53~71 111xxxx 111學測題
72~91 112xxxx 112學測題
92~112 110xxxx  110學測題
"""
header={
    0:"108",
    19:"107",
    34:"109",
    53:"111",
    72:"112",
    92:"110"
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


    