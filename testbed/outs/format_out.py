import re

pattern1 = re.compile(r'Occupy (edge|local) cpu time with \[(\d+)\] seconds')
pattern2 = re.compile(r'.mp4 with \[(\d+)\] seconds.')


file = open('client_7.out')

result = ''
line = file.readline()
while line:
    result += line
    line = file.readline()


result1 = pattern1.findall(result)
result2 = pattern2.findall(result)


for i in range(23):
    if result1[i][0] == 'edge':
        print(i, 0, result1[i][1], result2[i])
    else:
        print(i, result1[i][1], 0, result2[i])


