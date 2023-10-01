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
