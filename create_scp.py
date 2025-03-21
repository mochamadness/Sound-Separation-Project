import os

train_mix_scp = 'tr_mix.scp'
train_s1_scp = 'tr_s1.scp'
train_s2_scp = 'tr_s2.scp'

test_mix_scp = 'tt_mix.scp'
test_s1_scp = 'tt_s1.scp'
test_s2_scp = 'tt_s2.scp'

train_mix = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/train/mixed'
train_s1 = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/train/1'
train_s2 = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/train/2'

test_mix = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/test/mixed'
test_s1 = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/test/1'
test_s2 = 'D:/Projects/Conv-TasNet/Conv_TasNet_Pytorch/dataset/cleaned/chunks/test/2'

if not os.path.isdir(train_mix):
    print(f"Directory {train_mix} does not exist.")
else:
    with open(train_mix_scp, 'w') as tr_mix:
        for root, dirs, files in os.walk(train_mix):
            files.sort()
            for file in files:
                file_path = os.path.join(root, file)
                tr_mix.write(file + " " + file_path + '\n')
                print(f"Writing: {file} {file_path}")
    print(f"File {train_mix_scp} has been written successfully.")


tr_s1 = open(train_s1_scp,'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file+" "+root+'/'+file)
        tr_s1.write('\n')


tr_s2 = open(train_s2_scp,'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file+" "+root+'/'+file)
        tr_s2.write('\n')



tt_mix = open(test_mix_scp,'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file+" "+root+'/'+file)
        tt_mix.write('\n')


tt_s1 = open(test_s1_scp,'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file+" "+root+'/'+file)
        tt_s1.write('\n')


tt_s2 = open(test_s2_scp,'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file+" "+root+'/'+file)
        tt_s2.write('\n')

cv_mix_scp = 'cv_mix.scp'
cv_s1_scp = 'cv_s1.scp'
cv_s2_scp = 'cv_s2.scp'

cv_mix = '/dataset/cleaned/chunks/test/mixed'
cv_s1 = '/dataset/cleaned/chunks/test/1'
cv_s2 = '/dataset/cleaned/chunks/test/2'

cv_mix_file = open(cv_mix_scp,'w')
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        cv_mix_file.write(file+" "+root+'/'+file)
        cv_mix_file.write('\n')


cv_s1_file = open(cv_s1_scp,'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file+" "+root+'/'+file)
        cv_s1_file.write('\n')


cv_s2_file = open(cv_s2_scp,'w')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file+" "+root+'/'+file)
        cv_s2_file.write('\n')