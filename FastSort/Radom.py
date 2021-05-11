import random

Size = input('input the size of number,use "r" to use random.\t')
Size = random.randint(50, 100) if Size == "r" else Size
print(f"Size is {Size}")
MaxNum = input('input the max size of number,use "r" to use random.\t')
MaxNum = Size + random.randint(0, 100) if MaxNum == "r" else MaxNum
print(f"MaxNum is {MaxNum}")
MinNum = input('input the min size of number,use "r" to use random.\t')
MinNum = MaxNum - Size - random.randint(0, 100) if MinNum == "r" else MinNum
print(f"MinNum is {MinNum}")

num = [i for i in range(MinNum, MaxNum + 1)]
num = random.sample(num, Size)
random.shuffle(num)

txt = ""
for i in num:
    txt += f"{str(i)},"
txt=txt[:-1]

print(f"random list is : {txt}")

with open("input.txt", "w+", encoding="utf-8") as f:
    f.write(txt)
