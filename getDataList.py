import os
#coding=utf-8

trainDataPath1 = 'C:\\NDS\\traindata\\man_front'
trainDataPath2 = 'C:\\NDS\\traindata\\man_side'
trainDataPath3 = 'C:\\NDS\\traindata\\man_ride'
backgroundDataPath1 = 'C:\\NDS\\traindata\\background_front'
backgroundDataPath2 = 'C:\\NDS\\traindata\\background_side'
backgroundDataPath3 = 'C:\\NDS\\traindata\\background_ride'
hardDataPath1 = 'C:\\NDS\\traindata\\hard_front'
hardDataPath2 = 'C:\\NDS\\traindata\\hard_side'
hardDataPath3 = 'C:\\NDS\\traindata\\hard_ride'
file_objectNeg1s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallNegData1.txt', 'w')
file_objectNeg1m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleNegData1.txt', 'w')
file_objectNeg1l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeNegData1.txt', 'w')
file_objectTrain1s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallTrainData1.txt', 'w')
file_objectTrain1m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleTrainData1.txt', 'w')
file_objectTrain1l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeTrainData1.txt', 'w')
file_objectHard1s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallHard1.txt', 'w')
file_objectHard1m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleHard1.txt', 'w')
file_objectHard1l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeHard1.txt', 'w')

file_objectNeg2s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallNegData2.txt', 'w')
file_objectNeg2m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleNegData2.txt', 'w')
file_objectNeg2l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeNegData2.txt', 'w')
file_objectTrain2s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallTrainData2.txt', 'w')
file_objectTrain2m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleTrainData2.txt', 'w')
file_objectTrain2l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeTrainData2.txt', 'w')
file_objectHard2s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallHard2.txt', 'w')
file_objectHard2m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleHard2.txt', 'w')
file_objectHard2l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeHard2.txt', 'w')

file_objectNeg3s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallNegData3.txt', 'w')
file_objectNeg3m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleNegData3.txt', 'w')
file_objectNeg3l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeNegData3.txt', 'w')
file_objectTrain3s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallTrainData3.txt', 'w')
file_objectTrain3m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleTrainData3.txt', 'w')
file_objectTrain3l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeTrainData3.txt', 'w')
file_objectHard3s = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\SmallHard3.txt', 'w')
file_objectHard3m = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\MiddleHard3.txt', 'w')
file_objectHard3l = open('C:\\detectProject\\data\\sourceData\\TRAINDATA\\LargeHard3.txt', 'w')
smalln1, middlen1, largen1, smallt1, middlet1, larget1, smallh1, middleh1, largeh1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
smalln2, middlen2, largen2, smallt2, middlet2, larget2, smallh2, middleh2, largeh2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
smalln3, middlen3, largen3, smallt3, middlet3, larget3, smallh3, middleh3, largeh3 = 0, 0, 0, 0, 0, 0, 0, 0, 0

for root, dirs, files in os.walk(backgroundDataPath1):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectNeg1s.write(backgroundDataPath1 + "\\" + Path + '\n')
            smalln1 += 1
        elif str(Data)[0] == 'M':
            file_objectNeg1m.write(backgroundDataPath1 + "\\" + Path + '\n')
            middlen1 += 1
        elif str(Data)[0] == 'L':
            file_objectNeg1l.write(backgroundDataPath1 + "\\" + Path + '\n')
            largen1 += 1
for root, dirs, files in os.walk(backgroundDataPath2):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectNeg2s.write(backgroundDataPath2 + "\\" + Path + '\n')
            smalln2 += 1
        elif str(Data)[0] == 'M':
            file_objectNeg2m.write(backgroundDataPath2 + "\\" + Path + '\n')
            middlen2 += 1
        elif str(Data)[0] == 'L':
            file_objectNeg2l.write(backgroundDataPath2 + "\\" + Path + '\n')
            largen2 += 1
for root, dirs, files in os.walk(backgroundDataPath3):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectNeg3s.write(backgroundDataPath3 + "\\" + Path + '\n')
            smalln3 += 1
        elif str(Data)[0] == 'M':
            file_objectNeg3m.write(backgroundDataPath3 + "\\" + Path + '\n')
            middlen3 += 1
        elif str(Data)[0] == 'L':
            file_objectNeg3l.write(backgroundDataPath3 + "\\" + Path + '\n')
            largen3 += 1

for root, dirs, files in os.walk(trainDataPath1):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectTrain1s.write(trainDataPath1+"\\"+Path + '\n')
            smallt1 += 1
        elif str(Data)[0] == 'M':
            file_objectTrain1m.write(trainDataPath1+"\\"+Path + '\n')
            middlet1 += 1
        elif str(Data)[0] == 'L':
            file_objectTrain1l.write(trainDataPath1+"\\"+Path + '\n')
            larget1 += 1
for root, dirs, files in os.walk(trainDataPath2):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectTrain2s.write(trainDataPath2+"\\"+Path + '\n')
            smallt2 += 1
        elif str(Data)[0] == 'M':
            file_objectTrain2m.write(trainDataPath2+"\\"+Path + '\n')
            middlet2 += 1
        elif str(Data)[0] == 'L':
            file_objectTrain2l.write(trainDataPath2+"\\"+Path + '\n')
            larget2 += 1
for root, dirs, files in os.walk(trainDataPath3):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectTrain3s.write(trainDataPath3+"\\"+Path + '\n')
            smallt3 += 1
        elif str(Data)[0] == 'M':
            file_objectTrain3m.write(trainDataPath3+"\\"+Path + '\n')
            middlet3 += 1
        elif str(Data)[0] == 'L':
            file_objectTrain3l.write(trainDataPath3+"\\"+Path + '\n')
            larget3 += 1

for root, dirs, files in os.walk(hardDataPath1):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectHard1s.write(hardDataPath1+"\\"+Path + '\n')
            smallh1 += 1
        elif str(Data)[0] == 'M':
            file_objectHard1m.write(hardDataPath1+"\\"+Path + '\n')
            middleh1 += 1
        elif str(Data)[0] == 'L':
            file_objectHard1l.write(hardDataPath1+"\\"+Path + '\n')
            largeh1 += 1
for root, dirs, files in os.walk(hardDataPath2):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectHard2s.write(hardDataPath2 + "\\" + Path + '\n')
            smallh2 += 1
        elif str(Data)[0] == 'M':
            file_objectHard2m.write(hardDataPath2 + "\\" + Path + '\n')
            middleh2 += 1
        elif str(Data)[0] == 'L':
            file_objectHard2l.write(hardDataPath2 + "\\" + Path + '\n')
            largeh2 += 1
for root, dirs, files in os.walk(hardDataPath3):
    for Data in files:
        Path = str(Data)
        if str(Data)[0] == 'S':
            file_objectHard3s.write(hardDataPath3 + "\\" + Path + '\n')
            smallh3 += 1
        elif str(Data)[0] == 'M':
            file_objectHard3m.write(hardDataPath3 + "\\" + Path + '\n')
            middleh3 += 1
        elif str(Data)[0] == 'L':
            file_objectHard3l.write(hardDataPath3 + "\\" + Path + '\n')
            largeh3 += 1

file_objectTrain1s.close()
file_objectTrain2s.close()
file_objectTrain3s.close()
file_objectTrain1m.close()
file_objectTrain2m.close()
file_objectTrain3m.close()
file_objectTrain1l.close()
file_objectTrain2l.close()
file_objectTrain3l.close()
file_objectNeg1s.close()
file_objectNeg2s.close()
file_objectNeg3s.close()
file_objectNeg1m.close()
file_objectNeg2m.close()
file_objectNeg3m.close()
file_objectNeg1l.close()
file_objectNeg2l.close()
file_objectNeg3l.close()
file_objectHard1s.close()
file_objectHard2s.close()
file_objectHard3s.close()
file_objectHard1m.close()
file_objectHard2m.close()
file_objectHard3m.close()
file_objectHard1l.close()
file_objectHard2l.close()
file_objectHard3l.close()

lines=[]
f=open('D:\\projects\\pedestrianDec\\pedestrianDec\\ndsconfig.txt','r')
for line in f:
    lines.append(line)
f.close()
lines.__delslice__(15,24)
lines.insert(15,str(largeh1)+"\n")
lines.insert(15,str(largen1)+"\n")
lines.insert(15,str(larget1)+"\n")
lines.insert(15,str(middleh1)+"\n")
lines.insert(15,str(middlen1)+"\n")
lines.insert(15,str(middlet1)+"\n")
lines.insert(15,str(smallh1)+"\n")
lines.insert(15,str(smalln1)+"\n")
lines.insert(15,str(smallt1)+"\n")
lines.__delslice__(39,48)
lines.insert(39,str(largeh2)+"\n")
lines.insert(39,str(largen2)+"\n")
lines.insert(39,str(larget2)+"\n")
lines.insert(39,str(middleh2)+"\n")
lines.insert(39,str(middlen2)+"\n")
lines.insert(39,str(middlet2)+"\n")
lines.insert(39,str(smallh2)+"\n")
lines.insert(39,str(smalln2)+"\n")
lines.insert(39,str(smallt2)+"\n")
lines.__delslice__(63,72)
lines.insert(63,str(largeh3)+"\n")
lines.insert(63,str(largen3)+"\n")
lines.insert(63,str(larget3)+"\n")
lines.insert(63,str(middleh3)+"\n")
lines.insert(63,str(middlen3)+"\n")
lines.insert(63,str(middlet3)+"\n")
lines.insert(63,str(smallh3)+"\n")
lines.insert(63,str(smalln3)+"\n")
lines.insert(63,str(smallt3)+"\n")
s=''.join(lines)
f=open('D:\\projects\\pedestrianDec\\pedestrianDec\\ndsconfig.txt','w+')
f.write(s)
f.close()

print 'front:'
print 'Train:'
print 'small: ' + str(smallt1)
print 'middle: ' + str(middlet1)
print 'large: ' + str(larget1)
print 'Neg:'
print 'small: ' + str(smalln1)
print 'middle: ' + str(middlen1)
print 'large: ' + str(largen1)
print 'Hard:'
print 'small: ' + str(smallh1)
print 'middle: ' + str(middleh1)
print 'large: ' + str(largeh1)
print 'side:'
print 'Train:'
print 'small: ' + str(smallt2)
print 'middle: ' + str(middlet2)
print 'large: ' + str(larget2)
print 'Neg:'
print 'small: ' + str(smalln2)
print 'middle: ' + str(middlen2)
print 'large: ' + str(largen2)
print 'Hard:'
print 'small: ' + str(smallh2)
print 'middle: ' + str(middleh2)
print 'large: ' + str(largeh2)
print 'ride:'
print 'Train:'
print 'small: ' + str(smallt3)
print 'middle: ' + str(middlet3)
print 'large: ' + str(larget3)
print 'Neg:'
print 'small: ' + str(smalln3)
print 'middle: ' + str(middlen3)
print 'large: ' + str(largen3)
print 'Hard:'
print 'small: ' + str(smallh3)
print 'middle: ' + str(middleh3)
print 'large: ' + str(largeh3)
