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
먼저, $a$, $2a$, $3a$, $\cdots$, $\left(p-1\right)a$ 중 어느 것도 $\bmod p$
