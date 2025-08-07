from collections import deque
import sys

def solve():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))

    # Строим граф на N+1 вершинах (префиксные суммы S[0]…S[N])
    adj = [[] for _ in range(N+1)]
    for _ in range(Q):
        l = int(next(it))
        r = int(next(it))
        u = l - 1
        v = r
        adj[u].append(v)
        adj[v].append(u)

    # BFS от вершины 0 до вершины N
    dist = [-1] * (N+1)
    dist[0] = 0
    dq = deque([0])

    while dq:
        u = dq.popleft()
        if u == N:
            break
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)

    # Выводим результат
    if dist[N] == -1:
        print("No")
    else:
        print("Yes")
        print(dist[N])

if __name__ == "__main__":
    solve()
