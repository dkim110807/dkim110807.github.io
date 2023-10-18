---
title: Matching in General Graph
author: dkim110807
date: 2023-10-18 19:30:00 +0900
categories: [Algorithms, Graphs]
tags: [graphs]
math: true
---

## Matching
주어진 그래프 $G = \left(V, E\right)$의 부분 그래프 중 모든 노드의 차수가 $1$ 이하인 것을 의미한다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-Matching-Example.png){: width="572" height="489" }

### Maximum Matching
주어진 그래프 $G$의 매칭 중 간선의 수가 최대인 매칭을 의미한다. 이 경우는 아래의 가중치가 모든 같은 경우와 같다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-Maximum-Matching-Example.png){: width="272" height="289" }

### Maximum Matching in Weighted Graph
주어진 그래프 $G$가 가중치가 있는 경우 매칭 중 간선의 가중치가 최대인 매칭을 의미한다.

## Bipartite Graph
그래프의 정점을 두 그룹으로 적절히 분류하였을때 간선이 서로 다른 그룹의 정점을 연결하고 같은 그룹의 정점 사이 간선이 없는 그래프이다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-Bipartite-Graph.png){: width="472" height="489" }
