""" The function find_mismatched finds the mismatched parts in the generated responses by the LLM"""


def find_mismatched(l1, l2):
    n = len(l1)
    m = len(l2)

    dp = [([0] * m).copy() for _ in range(n)]
    par = [([0] * m).copy() for _ in range(n)]

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if i == n - 1 and j == m - 1:
                dp[i][j] = 0
            elif i == n - 1:
                dp[i][j] = dp[i][j + 1] + 1
                par[i][j] = 2
            elif j == m - 1:
                dp[i][j] = dp[i + 1][j] + 1
                par[i][j] = 1
            else:
                dp[i][j] = dp[i + 1][j] + 1
                par[i][j] = 1
                if dp[i][j] > dp[i][j + 1] + 1:
                    dp[i][j] = dp[i][j + 1] + 1
                    par[i][j] = 2
                if (l1[i] == l2[j]) and (dp[i][j] > dp[i + 1][j + 1]):
                    dp[i][j] = dp[i + 1][j + 1]
                    par[i][j] = 0

    def get_path(i, j):
        if i == n - 1 and j == m - 1:
            return [], [], []
        res = []
        add_ = []
        remove_ = []
        if par[i][j] == 0:
            res.append(f"({i}, {j}) --- Just go next {l1[i]}, and {l2[j]} are equal")
            x1, x2, x3 = get_path(i + 1, j + 1)
            res += x1
            add_ += x2
            remove_ += x3
        if par[i][j] == 1:
            res.append(f"({i}, {j}) --- Add {l1[i]} from l1 to l2 and go next.")
            add_.append((i, j))
            x1, x2, x3 = get_path(i + 1, j)
            res += x1
            add_ += x2
            remove_ += x3
        if par[i][j] == 2:
            res.append(f"({i}, {j}) --- Remove {l2[j]} from l2 and go next.")
            remove_.append(j)
            x1, x2, x3 = get_path(i, j + 1)
            res += x1
            remove_ += x3
            add_ += x2

        return res, add_, remove_

    return dp[0][0], get_path(0, 0)


l1 = [1, 2, 3, 4, 5, 6, 7]
l2 = [1, 2, 44, 3, 4, 60, 7]

x, y = find_mismatched(l1, l2)
print(x, y)
