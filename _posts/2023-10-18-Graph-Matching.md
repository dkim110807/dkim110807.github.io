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

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-1.png){: width="572" height="489" }

### Maximum Matching
주어진 그래프 $G$의 매칭 중 간선의 수가 최대인 매칭을 의미한다. 이 경우는 아래의 가중치가 모든 같은 경우와 같다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-2.png){: width="272" height="289" }

### Maximum Matching in Weighted Graph
주어진 그래프 $G$가 가중치가 있는 경우 매칭 중 간선의 가중치의 합이 최대인 매칭을 의미한다.

## Bipartite Graph
그래프의 정점을 두 그룹으로 적절히 분류하였을때 간선이 서로 다른 그룹의 정점을 연결하고 같은 그룹의 정점 사이 간선이 없는 그래프이다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-3.png){: width="472" height="489" }

## Maximum Matching in Bipartite Graph
이분 그래프에서의 최대 매칭은 최대 유량 문제로 해결 가능하다. 

1. 주어진 이분 그래프 $G = \left(A \cup B, E\right)$에 대해서 모든 간선에 대해 $A$에서 $B$로 가는 용량이 $1$인 간선을 만든다. 
2. 새로운 정점 $s$와 $t$를 추가하여, $s$와 $A$의 모든 정점을 용량이 $1$인 간선으로 이어주고, $t$와 $B$의 모든 정점을 용량이 $1$인 간선으로 이어준다. 
3. 이렇게 만들어진 새로운 그래프 $G^ \prime$에 대해 $s$에서 $t$로 유량을 흘려주면 최대 유량을 구성하는 간선이 최대 매칭에 대응된다.

### Proof
$M$을 위의 $G^\prime$에서 구한 최대 유량을 구성하는 간선 중 $A$와 $B$를 이어주는 간선들의 집합이라 한다면, 다음을 보이면 충분하다.

1. $M$은 가능한 매칭이다.
2. $M$은 가능한 매칭 중 간선의 수가 최대이다.

### $M$ is a matching
그래프 $G^\prime$을 정의한 방식에 의해 최대 유량을 구성하는 간선에는 어떠한 $A$의 노드에서도 최대 $1$개의 간선이 나가고, 어떠한 $B$의 노드에서도 최대 $1$개의 간선이 들어온다. 
만약, 어떠한 $A$의 노드에서 나가는 간선이 $2$개 이상이라면, $s$에서 이 노드를 연결하는 간선의 용량이 $1$이므로 불가능하다. 비슷하게, 어떠한 $B$의 노드에 들어오는 간선이 $2$개 이상이라면, $t$에서 이 노드를 연결하는 간선의 용량이 $1$이므로 불가능하다.

![Desktop View](/assets/img/posts/2023-10-18-Graph-Matching-4.png){: width="472"}

## Maximum Matching in General Graph
