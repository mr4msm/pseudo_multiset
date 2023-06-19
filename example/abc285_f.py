from string import ascii_lowercase


class BinaryIndexedTree:

    def __init__(self, n=0, init=0, data=None):
        if data is None:
            self.data = [init] * n
        else:
            self.data = list(data)

        self.tree = self.data_to_tree(self.data)
        self.n = len(self.data)
        if self.n == 0:
            self.pow2_le_n = 0
        else:
            self.pow2_le_n = 1 << (self.n.bit_length() - 1)

    @staticmethod
    def data_to_tree(data):
        n = len(data)
        tree = [0] * n
        s = [0] * (n + 1)
        for i in range(1, n + 1):
            s[i] = s[i - 1] + data[i - 1]
            tree[i - 1] = s[i] - s[i - (i & -i)]

        return tree

    def add(self, i, x):
        if i < 0 or i >= self.n:
            raise IndexError(f'{i} is out of range')

        self.data[i] += x

        i += 1
        while i <= self.n:
            self.tree[i - 1] += x
            i += (i & -i)

    def sum(self, right=None):
        """sum of [0, right)."""
        if right is None:
            right = self.n

        i = min(right, self.n)
        s = 0
        while i > 0:
            s += self.tree[i - 1]
            i -= (i & -i)

        return s

    def interval_sum(self, left, right=None):
        """sum of [left, right)."""
        return self.sum(right) - self.sum(left)

    def lower_bound(self, k):
        if k <= 0:
            return self.n

        idx = 0
        pow2 = self.pow2_le_n
        while pow2 > 0:
            if idx + pow2 <= self.n and self.tree[idx + pow2 - 1] < k:
                k -= self.tree[idx + pow2 - 1]
                idx += pow2
            pow2 >>= 1

        return idx

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, x):
        self.add(i, x - self.data[i])

    def __repr__(self):
        return f'{self.__class__.__name__}(data={self.data})'


N = int(input())
S = input()
Q = int(input())

queries = []
for _ in range(Q):
    queries.append(input().split())

bits = {c: BinaryIndexedTree(N) for c in ascii_lowercase}
for i in range(N):
    bits[S[i]].add(i, 1)

lt = BinaryIndexedTree(N - 1)
for i in range(N - 1):
    if S[i] > S[i + 1]:
        lt.add(i, 1)

s = [c for c in S]
for query in queries:
    if query[0] == '1':
        x, c = query[1:]
        x = int(x) - 1

        if x > 0:
            prev_lt = s[x - 1] > s[x]
            curr_lt = s[x - 1] > c
            if prev_lt is not curr_lt:
                if prev_lt:
                    lt.add(x - 1, -1)
                else:
                    lt.add(x - 1, 1)

        if x < N - 1:
            prev_lt = s[x] > s[x + 1]
            curr_lt = c > s[x + 1]
            if prev_lt is not curr_lt:
                if prev_lt:
                    lt.add(x, -1)
                else:
                    lt.add(x, 1)

        bits[s[x]].add(x, -1)
        s[x] = c
        bits[c].add(x, 1)
    else:
        left, right = map(int, query[1:])
        left -= 1
        right -= 1

        if lt.interval_sum(left, right) > 0:
            print('No')
            continue

        count = {}
        min_ci = -1
        max_ci = -1
        for i, c in enumerate(ascii_lowercase):
            count[c] = bits[c].interval_sum(left, right + 1)
            if count[c] > 0:
                if min_ci == -1:
                    min_ci = i
                max_ci = i

        min_ci += 1
        max_ci -= 1
        for i in range(min_ci, max_ci + 1):
            c = ascii_lowercase[i]
            if count[c] != bits[c].sum():
                print('No')
                break
        else:
            print('Yes')
