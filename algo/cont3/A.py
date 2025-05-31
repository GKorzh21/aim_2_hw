import sys

input = sys.stdin.readline

def read_int():
    return int(input())

def invr():
    return(map(int,input().split()))

def insr():
    s = input()

    return (list(s[:len(s) - 1]))

def path_sum(n):
    total = 0
    while n > 0:
        total += n
        n = n // 2
    return total

t = read_int()

for _ in range(t):
    n = read_int()
    
    print(path_sum(n))