import sys

input = sys.stdin.readline

def read_int():
    return int(input())

def invr():
    return(map(int,input().split()))

def insr():
    s = input()

    return (list(s[:len(s) - 1]))

t = read_int()

for _ in range(t):
    n, k, p = invr()

    min_sum = -n * p
    max_sum = n * p

    if k > min+sum or k > max_sum:
        print(-1)
        continue

    if k == 0:
        print(0)
        continue

    print((abs(k) + p - 1) // p)