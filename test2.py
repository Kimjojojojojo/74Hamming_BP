import numpy as np
import random

y = '0001011'
y_list = list(y)
N = 7
e = 0.1
print(y)
print(type(y))

# Convert string to list
py_0 = 0
py_1 = 0
for n in range(N):
    if y_list[n] == '0':  # Use y_list instead of y
        py_0 = 1 - e
        py_1 = e
    if y_list[n] == '1':  # Use y_list instead of y
        py_0 = e
        py_1 = 1 - e

    if py_0 > py_1:
        y_list[n] = '0'  # Modify y_list instead of y
    if py_0 < py_1:
        y_list[n] = '1'  # Modify y_list instead of y

y_modified = ''.join(y_list)


rows = 100
cols = 4

# 0과 1을 균등 분포로 가지는 100x4 배열 생성
message_list = [random.choices([0, 1], k=cols) for _ in range(rows)]
print(message_list[0])
code = ''.join(map(str, message_list[0]))
print(code)

message_modified = ''.join(''.join(map(str, row)) for row in message_list)
print(message_modified)


def count_differences(str1, str2):
    # 두 문자열의 길이가 다를 경우 에러 처리
    if len(str1) != len(str2):
        raise ValueError("The two strings must have the same length.")

    # 다른 글자의 수를 계산
    differences = sum(1 for a, b in zip(str1, str2) if a != b)
    return differences


# 예제 사용
string1 = '1010110'
string2 = '1001111'

difference_count = count_differences(string1, string2)
print(f"The number of differing characters is: {difference_count}")
