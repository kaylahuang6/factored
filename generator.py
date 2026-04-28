##定義五個factor的生成規則
#讓每個factor隨時間演化
import numpy as np


def mess3_matrix(x=0.15, a=0.6):#定義一個函數，用參數x(混和程度),a(偏向哪個方向)來產生mess3的 transition matrices:
    b = (1 - a) / 2
    y = 1 - 2 * x
    ay, bx, by, ax = a * y, b * x, b * y, a * x

    return np.array([
        [[ay, bx, bx],
         [ax, by, bx],
         [ax, bx, by]],

        [[by, ax, bx],
         [bx, ay, bx],
         [bx, ax, by]],

        [[by, bx, ax],
         [bx, by, ax],
         [bx, bx, ay]],
    ], dtype=np.float64)#三個主要方向,9個狀態


def tom_quantum_matrix(alpha=1.0, beta=3.0):
    gamma2 = 1.0 / (4.0 * (alpha**2 + beta**2))
    common_diag = 1.0 / 4.0
    middle_diag = (alpha**2 - beta**2) * gamma2
    off_diag = 2.0 * alpha * beta * gamma2

    return np.array([
        [[common_diag, 0.0,       off_diag],
         [0.0,         middle_diag, 0.0],
         [off_diag,    0.0,       common_diag]],

        [[common_diag, 0.0,      -off_diag],
         [0.0,         middle_diag, 0.0],
         [-off_diag,   0.0,       common_diag]],

        [[common_diag,  off_diag, 0.0],
         [off_diag,     common_diag, 0.0],
         [0.0,          0.0,        middle_diag]],

        [[common_diag, -off_diag, 0.0],
         [-off_diag,    common_diag, 0.0],
         [0.0,          0.0,        middle_diag]],
    ], dtype=np.float64)


def dominant_right_eigvec(mat):#系統偏好的主方向
    vals, vecs = np.linalg.eig(mat)#vals是特徵值,vecs是對應的特徵向量
    idx = np.argmax(vals.real)
    v = vecs[:, idx].real#找到特徵值最大的特徵向量,這個向量就是r
    return v


def stationary_left_vec(mat):#找穩定後會停在哪種分布
    vals, vecs = np.linalg.eig(mat.T)
    idx = np.argmin(np.abs(vals.real - 1.0))#找特徵職離1最近的，因為穩態不管怎麼棟都是自己，所以特徵值應該是1
    v = vecs[:, idx].real#找到特徵值最接近1的特徵向量,這個向量就是s
    v = v / v.sum()
    return v


class FactorProcess:
    def __init__(self, T, is_ghmm=False):
        self.T = T
        self.V = T.shape[0]
        self.S = T.shape[1]
        self.sumT = T.sum(axis=0)#知道自己的規則

        if is_ghmm:
            self.r = dominant_right_eigvec(self.sumT)#找到系統偏好的主方向,這個向量就是r,GHMM的r是根據transition matrix算出來的
        else:
            self.r = np.ones(self.S, dtype=np.float64)#如果不是GHMM,就假設每個狀態對系統的貢獻一樣,所以r是全1的向量

        self.init = stationary_left_vec(self.sumT)

    def obs_probs(self, belief):#下一個觀察出現的機率是根據目前的belief和transition matrix算出來的,denom是normalization factor,確保機率加起來是1
        denom = belief @ self.r
        out = np.empty(self.V, dtype=np.float64)

        for x in range(self.V):
            out[x] = (belief @ self.T[x] @ self.r) / denom

        out = np.clip(out, 0, None)#有負數就變成0
        return out / out.sum()

    def update(self, belief, obs):
        new_belief = belief @ self.T[obs]
        z = new_belief @ self.r
        return new_belief / z

    def sample_obs(self, belief, rng):
        p = self.obs_probs(belief)
        return rng.choice(self.V, p=p)


RADICES = [3, 3, 3, 4, 4]
BASE_VOCAB = int(np.prod(RADICES))   # 432
BOS = BASE_VOCAB                     # 432
VOCAB_SIZE = BASE_VOCAB + 1          # 433


def pack_subtokens(subtokens):#故意讓transformer不要看到它本身的結構,把五個factor的狀態打包成一個token
    t = 0
    for x, base in zip(subtokens, RADICES):
        t = t * base + int(x)
    return t


def build_independent_factors():
    return [
        FactorProcess(mess3_matrix(0.15, 0.6), is_ghmm=False),
        FactorProcess(mess3_matrix(0.15, 0.6), is_ghmm=False),
        FactorProcess(mess3_matrix(0.15, 0.6), is_ghmm=False),
        FactorProcess(tom_quantum_matrix(1.0, 3.0), is_ghmm=True),
        FactorProcess(tom_quantum_matrix(1.0, 3.0), is_ghmm=True),
    ]


def generate_sequence(seq_len=9, seed=0):#打造sequence的主函數,每次生成一個序列,seq_len是序列長度,seed是隨機種子
    rng = np.random.default_rng(seed)
    factors = build_independent_factors()
    beliefs = [f.init.copy() for f in factors]

    tokens = [BOS]
    factor_beliefs = [np.concatenate([b.copy() for b in beliefs])]

    for _ in range(seq_len - 1):
        subtokens = [f.sample_obs(b, rng) for f, b in zip(factors, beliefs)]
        token = pack_subtokens(subtokens)
        tokens.append(token)

        beliefs = [f.update(b, x) for f, b, x in zip(factors, beliefs, subtokens)]
        factor_beliefs.append(np.concatenate([b.copy() for b in beliefs]))

    tokens = np.array(tokens, dtype=np.int64)
    factor_beliefs = np.array(factor_beliefs, dtype=np.float32)
    return tokens, factor_beliefs


def sample_batch_with_beliefs(batch_size=32, seq_len=9, seed=0):
    rng = np.random.default_rng(seed)

    X_list = []
    Y_list = []
    B_list = []

    for _ in range(batch_size):
        s = int(rng.integers(0, 1_000_000_000))
        tokens, beliefs = generate_sequence(seq_len=seq_len, seed=s)

        X_list.append(tokens[:-1])     # 長度 8
        Y_list.append(tokens[1:])      # 長度 8
        B_list.append(beliefs[:-1])    # 跟 X 對齊，shape = [8, 15]

    X = np.array(X_list, dtype=np.int64)
    Y = np.array(Y_list, dtype=np.int64)
    B = np.array(B_list, dtype=np.float32)

    return X, Y, B


def sample_batch(batch_size=32, seq_len=9, seed=0):
    X, Y, _ = sample_batch_with_beliefs(
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed
    )
    return X, Y


if __name__ == "__main__":
    tokens, beliefs = generate_sequence(seq_len=9, seed=42)
    print("tokens:", tokens)
    print("beliefs shape:", beliefs.shape)

    X, Y = sample_batch(batch_size=4, seq_len=9, seed=123)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    X2, Y2, B2 = sample_batch_with_beliefs(batch_size=4, seq_len=9, seed=123)
    print("X2 shape:", X2.shape)
    print("Y2 shape:", Y2.shape)
    print("B2 shape:", B2.shape)