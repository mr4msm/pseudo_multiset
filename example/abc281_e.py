class BinaryIndexedTree:

    def __init__(self, n=0, init=0, data=None):
        if data is None:
            self.data = [init] * n
        else:
            self.data = list(data)

        self.tree = self.data_to_tree(self.data)
        self.n = len(self.data)
        self.pow2_le_n = 2**(self.n.bit_length() - 1)

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

    def sum(self, i=None):
        if i is None:
            i = self.n

        i = min(i, self.n)
        s = 0
        while i > 0:
            s += self.tree[i - 1]
            i -= (i & -i)

        return s

    def interval_sum(self, a, b=None):
        return self.sum(b) - self.sum(a)

    def lower_bound(self, k):
        if k <= 0:
            return self.n

        idx = 0
        pow2 = self.pow2_le_n
        while pow2 > 0:
            if idx + pow2 <= self.n and self.tree[idx + pow2 - 1] < k:
                k -= self.tree[idx + pow2 - 1]
                idx += pow2
            pow2 //= 2

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

    def __init__(self, elements):
        if isinstance(elements, dict):
            self.elements = sorted(elements)
            data = [elements[key] for key in self.elements]
            self.bit = BinaryIndexedTree(data=data)
            self.n = sum(data)
        else:
            self.elements = sorted(set(elements))
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
    def min(self):
        return self.get_k_th_element(1)

    @property
    def max(self):
        return self.get_k_th_element(self.n)

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


N, M, K = map(int, input().split())
A = list(map(int, input().split()))

multiset_lower = PseudoMultiset(A)
multiset_upper = PseudoMultiset(A)

sorted_first = sorted(A[:M])
k_sum = 0
for a in sorted_first[:K]:
    multiset_lower.add(a)
    k_sum += a
for a in sorted_first[K:]:
    multiset_upper.add(a)

k_sum_list = [k_sum]

for i in range(M, N):
    if A[i - M] in multiset_upper:
        multiset_upper.sub(A[i - M])
    else:
        multiset_lower.sub(A[i - M])
        k_sum -= A[i - M]

    lower_max = multiset_lower.max
    if lower_max is None or A[i] < lower_max:
        multiset_lower.add(A[i])
        k_sum += A[i]
    else:
        multiset_upper.add(A[i])

    if len(multiset_lower) > K:
        lower_max = multiset_lower.max
        k_sum -= lower_max
        multiset_lower.sub(lower_max)
        multiset_upper.add(lower_max)
    elif len(multiset_lower) < K:
        upper_min = multiset_upper.min
        k_sum += upper_min
        multiset_lower.add(upper_min)
        multiset_upper.sub(upper_min)

    k_sum_list.append(k_sum)

print(*k_sum_list)
