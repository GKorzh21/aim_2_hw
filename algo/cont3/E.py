t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    if n == 1:
        print(1)
    else:
        if k >= n - 1:
            print(1)
        else:
            print(n)