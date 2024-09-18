---
title: Scheduling preemptive jobs on a single machine
author: dkim110807
date: 2023-12-22 10:48:00 +0900
categories: [Algorithms, Scheduling]
tags: [greedy]
math: true
---

아래는 Tian, Z., Ng, C.T. & Cheng, T.C.E.의 2006년 [논문](https://link.springer.com/article/10.1007/s10951-006-7039-6) 
An $O(n^2)$ algorithm for scheduling equal-length preemptive jobs on a single machine to minimize total tardiness 의 내용을 바탕으로 작성하였다.

## Introduction
한 번에 하나의 작업이 가능한 하나의 기계를 사용하여 $n$개의 작업이 주어진 집합 $N = \left \lbrace 1, 2, \cdots, n \right \rbrace $을 배정하는 것으로, 
작업 $i$는 $r_i$의 시간부터 작업이 가능하며, 작업에 $p_i$의 시간이 걸리고, 작업은 $d_i \left( \ge r_i \right)$의 시간까지는 완료해야한다. 각각의 $r_i$, $p_i$, $d_i$
는 모두 정수이며, 각 작업에 걸리는 시간은 동일하다. 기계는 작업을 번갈아가며 해도 되며(Preemption allowed), 적절히 작업 순서를 배정하여 총 지각도인
$\sum T_i$를 최소화해야 한다. 이때, $T_i$는 작업 $i$에 대한 지각도로 작업 $i$가 시간 $C_i$에 완료되었을때 $\text{max}\left \lbrace 0, C_{i} - d_{i} \right \rbrace$로 정의된다.

## Definition

### Definition 1.
작업이 가능한 시간에 어느 기계도 쉬고 있지 않는다면 이 스케줄링은 지연 없는(non-delay) 스케줄링이다.

지연 없는 스케줄링의 경우 일반적인 작업을 번갈아가며 해도 되는 스케줄링에서 이득이다.

### Definition 2.
블록(Block) $B$는 $B \subseteq N$

### Definition 3.
ERD 스케줄은 


## Problems
[BOJ - 26248](https://noj.am/26248)

