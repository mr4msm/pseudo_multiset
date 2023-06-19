from collections import Counter


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


class PseudoMultiset:

    def __init__(self, elements, sort=True):
        if isinstance(elements, dict):
            if sort:
                self.elements = sorted(elements)
            else:
                self.elements = list(elements)
            data = [elements[key] for key in self.elements]
            self.bit = BinaryIndexedTree(data=data)
            self.n = self.bit.sum()
        else:
            if sort:
                self.elements = sorted(set(elements))
            else:
                self.elements = []
                ele_set = set()
                for ele in elements:
                    if ele not in ele_set:
                        ele_set.add(ele)
                        self.elements.append(ele)

            self.bit = BinaryIndexedTree(len(self.elements))
            self.n = 0

        self.element_to_idx = {e: i for i, e in enumerate(self.elements)}

    def add(self, x, n=1):
        if n < 0:
            self.sub(x, -n)
        else:
            self.bit.add(self.element_to_idx[x], n)
            self.n += n

    def sub(self, x, n=1):
        if n < 0:
            self.add(x, -n)
        else:
            idx = self.element_to_idx[x]
            n_in = self.bit[idx]
            if n_in > 0:
                n = min(n_in, n)
                self.bit.add(idx, -n)
                self.n -= n

    def discard(self, x):
        idx = self.element_to_idx[x]
        n_in = self.bit[idx]
        if n_in > 0:
            self.bit.add(idx, -n_in)
            self.n -= n_in

    def get_k_th_element(self, k, default=None):
        idx = self.bit.lower_bound(k)
        if idx == len(self.elements):
            return default
        else:
            return self.elements[idx]

    def sum_le(self, x):
        idx = self.element_to_idx[x]
        return self.bit.sum(idx + 1)

    def sum_lt(self, x):
        idx = self.element_to_idx[x]
        return self.bit.sum(idx)

    def max_le(self, x):
        return self.get_k_th_element(self.sum_le(x))

    def max_lt(self, x):
        return self.get_k_th_element(self.sum_lt(x))

    def min_ge(self, x):
        return self.get_k_th_element(self.sum_lt(x) + 1)

    def min_gt(self, x):
        return self.get_k_th_element(self.sum_le(x) + 1)

    @property
    def max(self):
        return self.get_k_th_element(self.n)

    @property
    def min(self):
        return self.get_k_th_element(1)

    def __len__(self):
        return self.n

    def __getitem__(self, x):
        return self.bit[self.element_to_idx[x]]

    def __setitem__(self, x, n):
        self.add(x, n - self[x])

    def __delitem__(self, x):
        self.discard(x)

    def __contains__(self, x):
        if self[x] > 0:
            return True
        else:
            return False

    def __repr__(self):
        elements = {e: d for e, d in zip(self.elements, self.bit.data)}
        return f'{self.__class__.__name__}({elements})'


N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

a_counter = Counter()
b_counter = Counter()

a_multiset = PseudoMultiset(range(N), sort=False)
b_multiset = PseudoMultiset(range(N), sort=False)
a_n_invs = 0
b_n_invs = 0

for i in range(N):
    a = A[i] - 1
    b = B[i] - 1
    a_counter[a] += 1
    b_counter[b] += 1
    a_n_invs += i - a_multiset.sum_le(a)
    b_n_invs += i - b_multiset.sum_le(b)
    a_multiset.add(a)
    b_multiset.add(b)

for i in range(N):
    if a_counter[i] != b_counter[i]:
        print('No')
        exit()

if a_n_invs % 2 != b_n_invs % 2:
    if max(a_counter.values()) >= 2:
        print('Yes')
    else:
        print('No')
else:
    print('Yes')
