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
    l, r, d, u = invr()

    if ((l**2 + u**2)**(1/2) == (d**2 + r**2)**(1/2)) and ((r**2 + u**2)**(1/2) == (d**2 + l**2)**(1/2)) and (u + d == l + r):
        print("YES")
    else:
        print("NO")