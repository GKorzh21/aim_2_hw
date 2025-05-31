import sys

input = sys.stdin.readline

def read_int():
    return int(input())

def invr():
    return(map(int,input().split()))

def insr():
    s = input()

    return (list(s[:len(s) - 1]))

def can_bob_reach_alice(n, s, a, b):
    s -= 1  # сделать индексацию с 0

    if a[0] == 0:
        return "NO"

    if a[s] == 1:
        return "YES"

    if b[s] == 0:
        return "NO"

    for i in range(s + 1, n):
        if a[i] == 1 and b[i] == 1:
            return "YES"

    return "NO"


n, s = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

print(can_bob_reach_alice(n, s, a, b))