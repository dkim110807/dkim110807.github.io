---
title: Miller - Rabin Primality Test
author: dkim110807
date: 2023-09-30 23:07:00 +0900
categories: [Algorithms, Math, Number Theory]
tags: [math, number theory, miller rabin]
math: true
---

## Fermat's Little Theorem

소수 $p$와 서로소인 $a$에 대해

$$ a^{p-1} \equiv 1 \pmod{p} $$

이 성립한다. <br>

### Proof.
먼저, $a, 2a, 3a, \cdots, \left(p-1\right)a$ 중 어느 것도 $\bmod p$에서 같지 않다는 것을 보이자.

서로 다른 두 정수 $i, j$에 대하여 $ia \equiv ja \pmod{p}$라면

$$ \left(i - j\right)a \equiv 0 \pmod{p} $$

$a$와 $p$가 서로소이므로 $i - j \equiv 0 \pmod{p}$이다. <br>
이때, $-p + 2 < i - j < p - 2$이므로 이러한 $i, j$는 존재하지 않는다.

위의 결과에 의해 $a, 2a, \cdots \left(p-1\right)a$는 $\bmod p$에서 $1, 2, \cdots, p - 1$의 값을 가질 수 있으므로,

$$ \left(p - 1\right)! a ^ {p - 1} \equiv \left(p - 1\right)! \pmod{p} $$

$\text{gcd}\left(\left(p - 1\right)!, p\right) = 1$이므로, 

$$ a ^ {p - 1} \equiv 1 \pmod{p} $$

### Corollary
소수 $p$와 임의의 자연수 $a$에 대하여

$$ a ^ {p} \equiv a \pmod{p} $$

이 성립한다.

## Fermat's Primality Test
판별하려는 수 $n$이 소수라면, 임의의 $n$과 서로소인 $a$에 대해서

$$ a ^ {n - 1} \equiv 1 \pmod{n} $$

이므로, 여러 $a$에 대해서 항상 성립하면 소수로 판별하는 방법이다.

### Carmichael Number
카마이클 수는 합성수 $n$에 대해 그보다 작고, $n$과 서로소인 임의의 $b$에 대해 항상 

$$b ^ {n - 1} \equiv 1 \pmod{n}$$

을 만족하는 수로 대표적으로 $341 = 11 \times 31$이 있다. 이러한 이유로 페르마 소수 판별법의 반례이기도 하다.

## Miller - Rabin Primality Test
먼저, 소수 중에 짝수는 $2$로 유일하므로 판별하려는 수는 $3$ 이상의 홀수라 가정하자. 이를 $p$라 하면, $p-1$은 짝수이므로 $p-1=2^{s}\cdot d$($s$는 홀수)로 표현할 수 있다. 페르마의 소정리에서 임의의 $p$와 서로소인 $a$에 대해

$$ a^{p-1} - 1 \equiv a^{2^{s} \cdot d} - 1 \equiv \left( a ^ {d} - 1 \right) \left(a^{d} + 1\right) \left(a^{2 \cdot d}+1\right) \cdots \left(a^{2^{s-1} \cdot d}+1\right) \equiv 0 \pmod{p} $$

$p$를 소수로 가정하였으므로 위의 인수분해한 결과 중 $0$이 존재해야 한다. $0$이 하나도 존재하지 않는다면 $p$가 소수라는 가정에 모순이므로 $p$는 합성수이다. 여러 $a$에 대해 이를 확인하면 합성수인데 소수로 잘못 판별될 확률을 줄일 수 있다.

실제로 판별하려는 수가 $4,759,123,141$보다 작다면 $a$로는 $2, 7, 61$을 사용하면 되고, $2^{63}-1$보다 작다면 $a$로는 $2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31$을 사용하면 소수라고 결정론적으로 말할 수 있다고 밝혀져 있다.

### Code

```cpp
class miller_rabin {
public:
    using u64 = uint64_t;

    bool is_prime(u64 n) {
        if (n < 2) return false;
        if (n == 2 || n == 3) return true;
        if (n % 6 != 1 && n % 6 != 5) return false;

        const auto &base = n < 4759123141ULL ? base_small : base_large;
        const int s = __builtin_ctzll(n - 1);
        const u64 d = n >> s;

        for (const auto &b: base) {
            if (b >= n) break;
            if (check_composite(n, b, d, s)) return false;
        }
        return true;
    }

protected:
    u64 mul(u64 a, u64 b, u64 m) {
        int64_t ret = a * b - m * u64(1.L / m * a * b);
        return ret + m * (ret < 0) - m * (ret >= int64_t(m));
    }

private:
    const std::vector<u64> base_small = {2, 7, 61}, base_large = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

    u64 pow(u64 a, u64 p, u64 m) {
        u64 ret = 1;
        for (; p; p >>= 1) {
            if (p & 1) ret = mul(ret, a, m);
            a = mul(a, a, m);
        }
        return ret;
    }

    bool check_composite(u64 n, u64 x, u64 d, int s) {
        x = pow(x, d, n);
        if (x == 1 || x == n - 1) return false;
        while (--s) {
            x = mul(x, x, n);
            if (x == n - 1) return false;
        }
        return true;
    };
};
```

### FFT
