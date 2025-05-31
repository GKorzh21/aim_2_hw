from collections import deque
import sys

def main():
    # Чтение ввода
    n, d = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()

    # Проверка на корректность ввода (по условию гарантируется, но лучше перестраховаться)
    if len(s) != n:
        print(-1)
        return

    # BFS
    visited = [False] * (n + 1)
    distance = [-1] * (n + 1)
    queue = deque()

    queue.append(1)
    visited[1] = True
    distance[1] = 0

    while queue:
        current = queue.popleft()
        if current == n:
            break
        for a in range(1, d + 1):
            next_pos = current + a
            if next_pos <= n and s[next_pos - 1] == '1' and not visited[next_pos]:
                visited[next_pos] = True
                distance[next_pos] = distance[current] + 1
                queue.append(next_pos)

    print(distance[n])

if __name__ == "__main__":
    main()