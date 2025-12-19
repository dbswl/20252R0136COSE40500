import sys
input = sys.stdin.readline

N, S = map(int, input().split())
series = list(map(int, input().split()))

def count_subseq_sum(idx, current_sum):
    if idx == N:
        return 1 if current_sum == S else 0
    return (count_subseq_sum(idx + 1, current_sum) +
            count_subseq_sum(idx + 1, current_sum + series[idx]))
answer = count_subseq_sum(0, 0)
print(answer if S != 0 else answer - 1)
