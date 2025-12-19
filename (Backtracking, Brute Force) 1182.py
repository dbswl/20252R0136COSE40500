import sys
input = sys.stdin.readline

N, S = map(int, input().split())
series = list(map(int, input().split()))
ans = 0


def find_sum(idx, curr_sum):
    global ans

    if idx >= N:
        return

    curr_sum += series[idx]
    if curr_sum == S:
        ans += 1

    find_sum(idx + 1, curr_sum - series[idx])
    find_sum(idx + 1, curr_sum)


find_sum(0, 0)
print(ans)
