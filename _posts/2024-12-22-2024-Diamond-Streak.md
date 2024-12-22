---
title: 2024 Diamond Streak
author: dkim110807
date: 2024-12-22 00:00:00 +0900
categories: [BOJ]
tags: [problem solving]
math: true
---

## Introduction

최근 10월 9일부터 하루에 다이아 1개 이상씩 푸는 도전을 진행했었다. 현재 목표는 올해까지는 채우는 것으로, 만약 된다면 100일은 넘기고 싶다. 여기에는 그동안 푼 다이아 이상인 문제들을 정리해보자고 한다. 

## 2024.10.09.

### <a href = "https://www.acmicpc.net/problem/14882">BOJ 14882</a>

문제는 다항식 $f(x)$가 주어질 때, $f(x_1), f(x_2), \cdots, f(x_m)$의 값을 구하라는 것이다. 주어진 모듈로가 $786,433 = 3 \times 2^{18} + 1$로 NTT-Friendly 소수이다. 즉, 단순하게 Multipoint Evaluation 하자. 정해는 NTT 3번 쓰는 것 같은데 잘 모르겠다.

## 2024.10.10.

### <a href = "https://www.acmicpc.net/problem/14878">BOJ 14878</a>

문제는 길이가 $n$인 수열 $A = \left \\{ a_1, a_2, \cdots, a_n \right \\}$이 주어질 때, $A$의 모든 부분 수열에 대해 각 부분 수열의 원소를 모두 xor한 값 중 가장 많이 등장한 값을 찾는 문제이다.

일단 기본적으로 fwht를 사용하자. 모든 부분 수열을 찾기 위해, xor 연산의 성질을 사용하자. $s_i = a_1 \oplus a_2 \oplus \cdots \oplus a_i$라 정의하자. 그러면, 부분 수열 $\left \\{ a_i, a_{i + 1}, \cdots, a_j \right\\}$의 xor 합은 $s_j \oplus s_{i - 1}$이 된다. 그리고 개수 또한 구해야 하므로 수열 $f$를 아래와 같이 정의하자.

$$f_{i} = \# \left\{ s_{j} = i \right\}$$

이제, $f$를 $f$와 fwht하자. 그러면 원하는 모든 부분 수열이 나타난다. 단, 길이가 $0$인 부분 수열은 허용하지 않기 때문에 값이 $0$인 경우는 주의하자.

```cpp
#include <bits/stdc++.h>

using ll = long long;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    std::vector<ll> f(1 << 16);
    f[0] += 1;
    for (int s = 0, x, i = 0; i < n; i++) {
        std::cin >> x;
        s ^= x;
        f[s] += 1;
    }

    auto fwht = [&](std::vector<ll> &a, bool inv = false) {
        int n = (int) a.size();

        for (int s = 2, h = 1; s <= n; s <<= 1, h <<= 1) {
            for (int l = 0; l < n; l += s) {
                for (int i = 0; i < h; i++) {
                    auto x = a[l + i], y = a[l + h + i];
                    a[l + i] = x + y;
                    a[l + h + i] = x - y;
                    if (inv) a[l + i] >>= 1, a[l + h + i] >>= 1;
                }
            }
        }
    };

    fwht(f);
    for (int i = 0; i < 1 << 16; i++) f[i] *= f[i];
    fwht(f, true);

    f[0] -= n;

    int max = 0;
    for (int i = 0; i < 1 << 16; i++) {
        if (f[i] > f[max]) max = i;
    }

    std::cout << max << " " << (f[max] >> 1);
}
```

## 2024.10.11.

### <a href = "https://www.acmicpc.net/problem/31986">BOJ 31986</a>

Golomb 수열의 $i$번째 항은 아래와 같다.

$$G_1 = 1, G_i = 1 + G_{i - G_{i - 1}}$$

또한, Golomb 수열의 정의에 따라 이는 수열에서 $i$가 등장하는 횟수이기도 하다. 또한, Golomb 수열은 단조증가수열이다. 우리는 마지막 수가 $m$으로 고정된 상황이므로, $m$보다 작은 수 $n - 1$개를 선택하는 가짓수를 구하면 된다.
아래와 같은 다항식을 생각해보자.

$$f(x) = \prod_{i = 1}^{m - 1} (G_i x + 1)$$

$f(x)$의 $x^{n - 1}$의 계수가 우리가 원하는 $m$보다 작은 수 $n - 1$개를 선택하는 가짓수가 된다. $f(x)$는 분할 정복을 사용하면 $\mathcal{O}(m \log^2 m)$에 계산 가능하다. 여기에 $G_m$을 곱하는 것을 잊지 말자. $m$이 작은 경우 주의하자.

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    if (m == 1) {
        std::cout << 1;
        return 0;
    }

    std::vector<int> G(m + 1, 1);
    for (int i = 2; i <= m; i++) G[i] = 1 + G[i - G[G[i - 1]]];

    std::queue<Polynomial<MInt>> queue;
    for (int i = 1; i < m; i++) queue.push({1, G[i]});

    while (queue.size() > 1) {
        auto a = queue.front();
        queue.pop();
        auto b = queue.front();
        queue.pop();
        queue.push(a * b);
    }

    std::cout << queue.front()[n - 1] * G[m];
}
```

## 2024.10.12.

### <a href = "https://www.acmicpc.net/problem/25662">BOJ 25662</a>

### <a href = "https://www.acmicpc.net/problem/31692">BOJ 31692</a>

## 2024.10.13.

### <a href = "https://www.acmicpc.net/problem/14424">BOJ 14424</a>

## 2024.10.14.

### <a href = "https://www.acmicpc.net/problem/21273">BOJ 21273</a>

## 2024.10.15.

### <a href = "https://www.acmicpc.net/problem/28182">BOJ 28182</a>

## 2024.10.16.

### <a href = "https://www.acmicpc.net/problem/23362">BOJ 23362</a>

## 2024.10.17.

### <a href = "https://www.acmicpc.net/problem/29150">BOJ 29150</a>

주어진 수열 $A = \left \\{ a_1, a_2, \cdots, a_n \right \\}$에 대해 다음 행렬의 행렬식을 구하라는 것이 문제다.

$$A = \begin{pmatrix}
a_{i} \choose {j - 1}
\end{pmatrix}_{i, j}$$

행렬식의 성질을 사용하면, 

$$\det A = \frac{1}{0! \cdot 1! \cdots (n - 1)!} \begin{vmatrix}
a_1 & a_1 (a_1 - 1) & \cdots & a_1 (a_1 - 1) \cdots (a_1 - n + 1) \\
a_2 & a_2 (a_2 - 1) & \cdots & a_2 (a_2 - 1) \cdots (a_2 - n + 1) \\
\vdots & \vdots & \ddots & \vdots \\
a_n & a_n (a_n - 1) & \cdots & a_n (a_n - 1) \cdots (a_n - n + 1)
\end{vmatrix}$$

기본 열 연산 사용해서 적절히 정리하면,

$$\det A = \frac{1}{0! \cdot 1! \cdots (n - 1)!} \begin{vmatrix}
a_1 & a_1^2 & \cdots & a_1^n \\
a_2 & a_2^2 & \cdots & a_2^n \\
\vdots & \vdots & \ddots & \vdots \\
a_n & a_n^2 & \cdots & a_n^n
\end{vmatrix}$$

방데르몽드 행렬식이다. 이는 $\mathcal{O}(n^2)$에 구현 가능하다.

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    std::vector<MInt> fact(1'001, 1), inv(1'001);
    for (int i = 1; i <= 1'000; i++) fact[i] = fact[i - 1] * i;
    inv[1'000] = fact[1'000].inv();
    for (int i = 1'000; i >= 1; i--) inv[i - 1] = inv[i] * i;

    int T;
    for (std::cin >> T; T--;) {
        int n;
        std::cin >> n;

        MInt det = 1;
        std::vector<ll> v(n);
        for (int i = 0; i < n; i++) {
            std::cin >> v[i];
            for (int j = 0; j < i; j++) det *= v[i] - v[j];
            det *= inv[i];
        }
        std::cout << det << "\n";
    }
}
```

## 2024.10.18.

### <a href = "https://www.acmicpc.net/problem/16844">BOJ 16844</a>

## 2024.10.19.

### <a href = "https://www.acmicpc.net/problem/12231">BOJ 12231</a>

## 2024.10.20.

### <a href = "https://www.acmicpc.net/problem/17635">BOJ 17635</a>

쿼리를 $B$개씩 묶어서 처리하자.

## 2024.10.21.

### <a href = "https://www.acmicpc.net/problem/31699">BOJ 31699</a>

## 2024.10.22.

### <a href = "https://www.acmicpc.net/problem/13727">BOJ 13727</a>

초항은 적절한 bit dp를 통해서 찾아줄 수 있다. 이후는 Berlekamp-massey를 사용하자. 초항 중 일부는 아래와 같다.

```text
1, 272, 589185, 930336768, 853401154, 217676188, 136558333, 415722813, 985269529, 791527976, 201836136, 382110354, 441223705, 661537677, 641601343, 897033284, 816519670, 365311407, 300643484, 936803543, 681929467, 462484986, 13900203, 657627114, 96637209, 577140657, 600647073, 254604056, 102389682, 811580173, 592550067, 587171680, 526467503, 265885773, 951722780, 219627841, 371508152, 283501391, 159234514, 439380999, 722868959, 125599834, 351398134, 456317548, 365496182, 614778702, 502680047, 193063685, 309004764, 743901785, 870955115, 312807829, 160375015, 691844624, 137034372, 350330868, 895680450, 282610535, 317897557, 28600551, 583305647, 539409363, 327406961, 627805385, 680183978, 681299085, 954964592, 743524009, 788048339, 699454626, 666369521, 857206425, 490463127, 477198247, 599963928, 21247982, 107843532, 753662937, 239039324, 608530376, 523383010, 654448101, 801430395, 393034561, 93313778, 983052766, 240336620, 825539982, 525118275, 563899476, 706271688, 547405697, 477082486, 664058071, 353207278, 729486413, 795704637, 999271072, 540749624, 411451016, 736422999, 879369181, 918733916, 982303557, 512499644, 261033810, 391766409, 334092786, 931794834, 854181848, 821090190, 751839258, 433126935, 571194155, 52438113, 552977155, 320805296, 173355929, 969659468, 258854248, 159509877, 374487748, 401382023, 44060530, 510164669, 336596764, 652050424, 373872552, 517226592, 719871041, 43959496, 235333335, 304962191, 253114421, 43638769, 361871585, 8060121, 147014624, 114846460, 430864038, 368951246, 863795701, 36066788, 971606149, 935875286, 486724123, 73790652, 236936530, 307697424, 753314001, 40450345, 529462842, 166162047, 974102330, 600865526, 63237062, 749041914, 670937123, 806399597, 776678839, 842565920, 608499253, 469062485, 842196981, 247762946, 778570576, 237951782, 286343384, 988318575, 147255879, 905747089, 711062313, 21396079, 826846622, 443781794, 786474911, 400737121, 844768961, 686214818, 590050845, 855473150, 18501778, 33258755, 398169058, 811192244, 710397887, 591757177, 775311969, 168256434, 509615161, 489764304, 605188191, 498085780, 164388183, 524662873, 322602324, 853641480, 205349527, 308211944, 93153206, 734257752, 68829302, 443687521, 524241394, 591557198, 308656747, 511733449, 943095360, 194572043, 420913382, 679842332, 684364764, 134540921, 551103000, 700528141, 54414645, 814404379, 3421752, 316740512, 853118601, 894201609, 877520795, 244106463, 358840411, 411662431, 953845173, 239397728, 391633640, 745859650, 6417562, 246353318, 900069523, 877218664, 234394818, 171521822, 184466314, 316351773, 353811494, 617940271, 731132804, 656046921, 2378554, 305082811, 860468755, 877839522, 884387573, 83314799, 753963703, 702751847, 739819061, 2908431, 897890934, 45761348, 828368065, 248920872, 715741260, 472582555, 257809564, 504265160, 212679404, 760999037, 933814419, 455144854, 115887513, 42779561, 979146543, 823633147, 661808844, 538301653, 428602586, 509439171, 479229429, 261257513, 999896005, 172280049, 953689482, 664590174, 996316893, 546207180, 707356501, 304296138, 533124440, 369260400, 490902637, 992556759, 637447097, 474452901, 504373318, 197786209, 110473647, 296172594, 37239342, 311853805, 143361842, 42640380, 937977495, 327898926, 514659535, 361824277, 544996090, 451220726, 824086130, 884955, 511443014, 395360539, 449898107, 223530402, 546772517, 258362063, 361824843, 703495720, 374810161, 671021377, 856505641, 310584879, 638513768, 983731358, 597122266, 213389863, 981491603, 575102614, 594784593, 292532508, 281245066, 634033678, 153303431, 963100023, 144037118, 6050496, 303134432, 474494809, 357253830, 166938350, 98207573, 615380032, 232487217, 285431788, 189692050, 907302321, 926078717, 8387032, 853492429, 726765260, 243883808, 395371645, 189821148, 874421195, 365623069, 194142517, 913761737, 766995838, 68222432, 672560338, 851998484, 456256902, 717471906, 915098281, 997175561, 534008631, 116790836, 439953630, 57551462, 39889129, 371089188, 551290687, 837754287, 908748672, 273868942, 313341018, 734515372, 64282311, 541356990, 264748394, 8138923, 521096093, 750390135, 371662374, 255683937, 290203432, 920075087, 585901851, 694431618, 258782031, 579548194, 931799539, 628120244, 103331049, 717965194, 680648992, 959546763, 223843862, 664707998, 736881370, 860882803, 52125159, 697817765, 985352364, 108984272, 519429710, 472730914, 239367114, 182848612, 918084646, 120715246, 228998083, 166126284, 896136381, 983683167, 708774238, 549425249, 273923925, 376526561, 397287999, 577108313, 593040262, 385928329, 10034007, 578269242, 606427925, 441991170, 47175256, 879533898, 775912030, 320916106, 908362087, 159421394, 264775867, 295780366, 673526561, 236297549, 229412443, 876524965, 125746838, 183319452, 942708529, 576345236, 686895476, 484308250, 169120748, 53287426, 250458519, 631376555, 559423076, 196713232, 462032976, 468521569, 441879142, 481315394, 190597184, 934735249, 583021521, 550357898, 536996179, 565566613, 987718257, 748557572, 350527748, 905143511, 631462405, 684093578, 567928603, 351909501, 417724763, 187412313, 92510936, 217689712, 813690295, 248912375, 806915101, 420571491, 886430898, 617302763, 72031433, 289695536, 655746239, 112476959, 584008263, 259643243, 954193800, 67910730, 838205549, 936433832, 212728810, 157018059, 576199395, 557382179, 291472517, 399299021, 413176768, 514128062, 839046181, 441888823, 938173391, 905971671, 601156430, 957449712, 122103172, 901927906, 350200325, 897365057, 310050187, 868063528, 674414585, 941092054, 742522475, 587193220, 913716314, 10544152, 808650487, 431760020, 306672990, 785879925, 711071061, 108011016, 457113343, 599958364, 929884100, 748947313, 808870615, 752079106, 492485605, 349155330, 322971962, 610218914, 364155566, 619765398, 707279011, 513197687, 219572609, 704604559, 519552244, 379970387, 892198502, 700506389, 314222809, 525439586, 121728573, 450533656, 861911131, 93317774, 856596631, 282316530, 568639914, 715114306, 851504952, 716647063, 389741386, 31494359, 412400492, 848142084, 323448999, 204589423, 694247623, 857921942, 447860627, 144948569, 816409913, 993524809, 528976150, 741686615, 570656962, 405627334, 803189581, 674203462, 18269388, 786432200, 295506914, 379311726, 604834489, 187916877, 687420066, 89481104, 111911146, 606431351, 381396394, 361397837, 57673117, 180450438, 865782200, 767056334, 126567355, 155856414, 363911840, 973336786, 191532090, 945199153, 865329660, 144517973, 743195079, 848362854, 378202740, 10195668, 729535192, 436732685, 304391694, 694263295, 106045215, 925259439, 182288043, 574796625, 869731473, 157104177, 956168826, 160388758, 117001724, 554862950, 267628755, 973142907, 237010326, 312897965, 594270485, 16307105, 222444246, 334942547, 253152565, 745209781, 966387248, 731165798, 758642349, 935354071, 441994761, 675729844, 930826497, 908401534, 664189452, 786830322, 333717452, 64907065, 262566848, 146787248, 9223216, 221375041, 230238737, 608560604, 132233448, 18734979, 663327095, 856125661, 486160929, 118742057, 275590854, 389712514, 919227980, 827818829, 226942941, 323763022, 616958281, 852052023, 359803209, 600899512, 709063838, 663418782, 903380427, 133292713, 790178962, 691486574, 534442697, 485410240, 267284103, 564988670, 44405640, 121210071, 536438535, 687636317, 270569595, 278427791, 256641871, 719219371, 992922472, 127648292, 621873342, 355602691, 54169398, 363156222, 623272255, 716605449, 281315687, 983959859, 809935173, 932854397, 595401871, 893975021, 8607470, 991360747, 70818834, 293868123, 593160163, 3799244, 507142608, 833309625, 73254107, 103949277, 431358806, 692186731, 890868965, 49753835, 93986623, 510111238, 643874705, 516264615, 22033933, 510943632, 49292255, 294496383, 923313926, 398888237, 845435861, 704618493, 141346807, 717000630, 82195913, 327832235, 763397316, 948481992, 281189585, 280507329, 100398690, 635655688, 191059163, 607177712, 401580964, 754949300, 553977767, 841567850, 15965206, 236689009, 848820181, 447806081, 61016339, 918781218, 951286024, 570690033, 619019577, 288381859, 149624974, 444646220, 4947729, 651640455, 370275187, 475289780, 133059502, 212789696, 793721625, 957726571, 522835170, 603837256, 653112692, 416839501, 409206469, 779810381, 706105560, 671459577, 289409893, 167766878, 678483948, 167782908, 465225200, 387567776, 927682744, 668862691, 156258774, 991347616, 234515040, 520346215, 34353653, 850275587, 342979587, 789216591, 815587275, 705618886, 891334966, 268831978, 833805992, 491183344, 694734814, 789571087, 881809425, 370848535, 108749455, 741083049, 287419320, 756289121, 784114620, 999173812, 79279205, 873455514, 888102477, 368261191, 746582965, 242288127, 687442916, 278149559, 72098982, 70686432, 193575425, 26564465, 381630782, 20785017, 33474769, 601743555, 785926208, 294248833, 894107185, 635607997, 16733708, 78039865, 790031523, 974963906, 816791371, 931916823, 141942464, 307080772, 794749341, 720585255, 736684920, 667705553, 135049124, 100924990, 671598653, 451732762, 652203338, 193586839, 509285048, 115842941, 435541455, 146759437, 552406968, 63381575, 16858045, 997450669, 271075012, 150519536, 25817820, 395292275, 271691540, 289322922, 953644328, 905177895, 981427722, 388338779, 979753012, 193827621, 133126598, 181759680, 327972210, 619987653, 876398482, 74281357, 954397912, 441830651, 495588011, 681708634, 151650234, 902194685, 449366976, 102484519, 610814753, 529627208, 196420443, 678713814, 606047193, 525492393, 840449363, 341065718, 947945379, 822478738, 373435945, 250226415, 588719763, 879416378, 559150560, 603624130, 709367474, 478264543, 163630088, 418455456, 212679988, 916813964, 26308579, 509836366, 456399104, 559386486, 734131041, 288173222, 299010613, 845000607, 880839071, 198281412, 154156747, 646166291, 800505402, 106233979, 868674372, 676613092, 939632584, 694374510, 395461660, 549043996, 268910536, 575558231, 609173979, 61849818, 172355235, 83558384, 433670211, 886836631, 189008228, 133562559, 539956146, 226089731, 692753973, 99259370, 939547649, 145199521, 927593133, 657190211, 136307085, 938662762, 856063055, 65589003, 179764018, 691936698, 750002497, 144369732, 173951433, 21118633, 778105146, 494919594, 142318419, 557544107, 382910573, 645068546, 491476468, 408961131, 646334263, 119898816, 32635757, 933939976, 891122285, 42029420, 178250754, 655047369, 644214377, 967148218, 470718604, 183618639, 865789294, 517974864, 399035837, 705628662, 117686611, 814967715, 598820747, 570054794, 735890423, 22591401, 960388621, 124930416, 712352602, 229934844, 93688683, 628641070, 570254384, 480748730, 53592426, 136726216, 649304109, 243918630, 464889944, 731766745, 120548153, 828923657, 642197857, 592638490, 371851039, 655454755, 908490432, 68189233, 61120097, 593761802, 218143256, 727260927, 284037501, 316780286, 237874586, 462723772, 134293351, 241344603, 958109243, 489548112, 744994810, 542716530, 733055247, 240186696, 588039765, 155910211, 360033424, 910473602
```

### <a href = "https://www.acmicpc.net/problem/27520">BOJ 27520</a>

제주도 문제와 비슷하다. 원하는 답 $r$을 이분 탐색으로 구하자. 경찰서의 위치는 각 도로에서 최대 $r$만큼 떨어져 있으므로, 각 도로에서 거리가 $+r$만큼 떨어진 하나의 직선과 $-r$만큼 떨어진 다른 직선 사이에 위치해야한다. 각각은 반평면이므로 반평면 교집합으로 영역을 구할 수 있고, 따라서 전체 직선에 대해서 모두 반평면을 구한 다음에 반평면 교집합이 존재하는지로 결정 가능하다. 시간 초과와 틀렸습니다 사이에서 적절히 이분 탐색을 해주자.

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    std::vector<std::array<long double, 3>> v(n);
    for (auto &[x, y, z]: v) std::cin >> x >> y >> z;

    auto f = [&](long double r) -> bool {
        std::vector<Halfplane> half;
        for (int i = 0; i < n; i++) {
            Point a, b;
            if (v[i][0] == 0) {
                a = Point{0, -v[i][2] / v[i][1]};
                b = Point{1, -v[i][2] / v[i][1]};
            } else if (v[i][1] == 0) {
                a = Point{-v[i][2] / v[i][0], 0};
                b = Point{-v[i][2] / v[i][0], 1};
            } else {
                a = Point{0, -v[i][2] / v[i][1]};
                b = Point{-v[i][2] / v[i][0], 0};
            }
            if (a.x > b.x) std::swap(a, b);
            half.emplace_back(a, b), half.back().move(-r);
            half.emplace_back(b, a), half.back().move(-r);
        }
        return !(intersect(half).empty());
    };

    long double low = 0, high = 1e7;
    for (int rep = 100; rep--; ) {
        long double mid = (low + high) / 2.;
        if (f(mid)) high = mid;
        else low = mid;
    }
    std::cout << std::setprecision(20) << std::fixed << low;
}
```

### <a href = "https://www.acmicpc.net/problem/14854">BOJ 14854</a>

원하는 모듈로 $142857 = 3^3 \cdot 11 \cdot 13 \cdot 37$로 $3^3$이 문제다. 만약, $n \choose r$의 값을 $3^3 = 27$로 나눈 나머지를 구할 수 있다면 나머지는 중국인의 나머지 정리로 가능하다.

Todo.

### <a href = "https://www.acmicpc.net/problem/14131">BOJ 14131</a>

정해는 아니지만 간단하게 Splay Tree로 구현 가능하다. 구간 뒤집기 쿼리만 구현하면 된다.

## 2024.10.23.

### <a href = "https://www.acmicpc.net/problem/14853">BOJ 14853</a>

## 2024.10.24.

### <a href = "https://www.acmicpc.net/problem/31272">BOJ 31272</a>

## 2024.10.25.

### <a href = "https://www.acmicpc.net/problem/1616">BOJ 1616</a>

문제의 답은 de Bruijn 수열이다. 또는 <a href = "https://www.acmicpc.net/problem/28077">사탕 팔찌</a> 문제처럼 처리하면 된다. 이 방법은 최적화를 많이 해야 한다고 한다.

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int k, m;
    std::cin >> k >> m;

    std::vector<int> ans;
    auto f = [&](auto &&f, int t, int p) -> void {
        if (t > m) {
            if (m % p == 0) {
                for (int i = 1; i < p + 1; i++) ans.push_back(a[i]);
            }
        } else {
            a[t] = a[t - p];
            f(f, t + 1, p);
            for (int i = a[t - p] + 1; i < k; i++) {
                a[t] = i;
                f(f, t + 1, t);
            }
        }
    };
    f(f, 1, 1);

    for (auto &x: ans) std::cout << x << " ";
}
```

## 2024.10.26.

### <a href = "https://www.acmicpc.net/problem/1763">BOJ 1763</a>

## 2024.10.27.

### <a href = "https://www.acmicpc.net/problem/3906">BOJ 3906</a>

문제를 해석해보면, 다각형이 star-shaped가 되기 위해서는 다각형의 모서리를 직선으로 생각했을 때, 이들의 반평면 교집합이 존재하면 된다. 문제의 범위도 작은데 왜 D3인지는 잘...

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    for (int n;;) {
        std::cin >> n;

        if (n == 0) break;

        std::vector<Point> v(n);
        for (auto &[x, y]: v) std::cin >> x >> y;

        std::vector<Halfplane> hp;
        for (int i = 0; i < n; i++) {
            hp.emplace_back(v[i], v[(i + 1) % n]);
        }

        std::cout << !intersect(hp).empty() << "\n";
    }
}
```

## 2024.10.28.

### <a href = "https://www.acmicpc.net/problem/16857">BOJ 16857</a>

## 2024.10.29.

### <a href = "https://www.acmicpc.net/problem/18939">BOJ 18939</a>

## 2024.10.30.

### <a href = "https://www.acmicpc.net/problem/16127">BOJ 16127</a>

미생물을 만드는 2가지 방법 중에서, $i$번 미생물을 구입하여 넣는 방법을 어떠한 $root = n + 1$번 미생물에서 약물을 사용하여 $i$번 미생물 하나를 만든다고 생각하자. 또한, 문제 조건 상에서 각각의 미생물을 최소 하나씩은 만들어야 함을 알 수 있다. 그러면, 이제 각각의 미생물을 하나씩 만드는 최소 비용을 생각해보자.

문제의 상황은 하나의 방향 그래프로 생각 가능하다. $i$번 미생물로부터 $j$번 미생물로 비용 $c$로 만들 수 있다면 그래프에서 $i \rightarrow j$ 방향 간선의 가중치 $c$를 추가하자. 그러면, 먼저 $root$ 노드에서 다른 모든 노드로 갈 수 있으며, 하나씩 최소 비용으로 만들기 때문에 여기서 directed mst를 만든다고 생각할 수 있다. 이제 하나씩 모든 미생물을 만들었으므로 나머지 $z_i - 1$개의 미생물은 단순하게 이 미생물을 만드는 최소 비용으로 만들면 된다.

```cpp
#include <bits/stdc++.h>

using ll = long long;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    std::vector<int> x(n + 1);
    for (int i = 1; i <= n; i++) std::cin >> x[i];

    std::vector<ll> min(n + 1);
    std::vector<std::array<ll, 3>> edge;
    for (int i = 1; i <= n; i++) std::cin >> min[i], edge.push_back({n + 1, i, min[i]});

    for (int i = 1; i <= n; i++) {
        for (ll z, j = 1; j <= n; j++) {
            std::cin >> z;
            edge.push_back({i, j, z}), min[j] = std::min(min[j], z);
        }
    }

    ll ans = 0;
    {
        int root = n + 1, m = n + 1;
        while (true) {
            std::vector<std::array<ll, 2>> delta(m + 1, {INT64_MAX, root});
            for (const auto &[u, v, c]: edge) {
                if (u != v && c < delta[v][0]) {  // u -> v
                    delta[v][0] = c, delta[v][1] = u;
                }
            }
            delta[root][0] = 0;    // root
            for (int i = 1; i <= m; i++) ans += delta[i][0];

            int cnt = 0;
            std::vector<int> cycle(m + 1, -1);
            cycle[root] = ++cnt;
            for (int i = 1; i <= m; i++) {
                if (cycle[i] != -1) continue;
                int v = i;
                for (; cycle[v] == -1; v = delta[v][1]) cycle[v] = -2;
                if (cycle[v] == -2) {
                    cnt += 1;
                    for (; cycle[v] == -2; v = delta[v][1]) cycle[v] = cnt;
                }
                for (v = i; cycle[v] == -2; v = delta[v][1]) cycle[v] = ++cnt;
            }
            if (cnt == m) break;

            m = cnt;
            for (auto &[u, v, c]: edge) {
                c -= delta[v][0], u = cycle[u], v = cycle[v];
            }
            root = cycle[root];
        }
    }

    for (int i = 1; i <= n; i++) ans += 1ll * min[i] * (x[i] - 1);

    std::cout << ans;
}
```

## 2024.10.31.

### <a href = "https://www.acmicpc.net/problem/25839">BOJ 25839</a>

이 문제 역시 directed mst를 구하는 문제이다. 이분 탐색을 사용하자. 문제 조건에서 전체 그래프가 root에서 각 정점으로 모두 갈 수 있음이 보장되므로 간단하게 구현할 수 있다.

```cpp
#include <bits/stdc++.h>

using ll = long long;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    long double limit = 0;
    std::cin >> limit;

    int n, k;
    std::cin >> n >> k;

    std::vector<std::array<int, 4>> edges;
    for (int a, b, c, d; k--; ) {
        std::cin >> a >> b >> c >> d;
        edges.push_back({b, a, c, d});
    }

    auto f = [&](long double x) -> bool {
        long double sum = 0;
        int root = 1, m = n;
        std::vector<std::tuple<int, int, long double>> edge;
        for (auto &[a, b, c, d]: edges) edge.emplace_back(a, b, c + d * x);
        while (true) {
            std::vector<std::pair<long double, int>> delta(m + 1, {1e10, root});
            for (const auto &[u, v, c]: edge) {
                if (u != v && c < delta[v].first) {  // u -> v
                    delta[v].first = c, delta[v].second = u;
                }
            }
            delta[root].first = 0;    // root
            for (int i = 1; i <= m; i++) sum += delta[i].first;

            int cnt = 0;
            std::vector<int> cycle(m + 1, -1);
            cycle[root] = ++cnt;
            for (int i = 1; i <= m; i++) {
                if (cycle[i] != -1) continue;
                int v = i;
                for (; cycle[v] == -1; v = delta[v].second) cycle[v] = -2;
                if (cycle[v] == -2) {
                    cnt += 1;
                    for (; cycle[v] == -2; v = delta[v].second) cycle[v] = cnt;
                }
                for (v = i; cycle[v] == -2; v = delta[v].second) cycle[v] = ++cnt;
            }
            if (cnt == m) break;

            m = cnt;
            for (auto &[u, v, c]: edge) {
                c -= delta[v].first, u = cycle[u], v = cycle[v];
            }
            root = cycle[root];
        }

        return sum + x <= limit;
    };

    long double low = 0, high = limit, ans = 0;
    for (int rep = 100; rep--; ) {
        long double mid = (low + high) / 2.;
        if (f(mid)) ans = mid, low = mid;
        else high = mid;
    }

    std::cout << std::fixed << std::setprecision(20) << ans;
}
```

## 2024.11.01.

### <a href = "https://www.acmicpc.net/problem/7907">BOJ 7907</a>

## 2024.11.02.

### <a href = "https://www.acmicpc.net/problem/14960">BOJ 14960</a>

### <a href = "https://www.acmicpc.net/problem/17625">BOJ 17625</a>

### <a href = "https://www.acmicpc.net/problem/23863">BOJ 23863</a>

위와 동일한 문제이다.

## 2024.11.03.

### <a href = "https://www.acmicpc.net/problem/7057">BOJ 7057</a>

## 2024.11.04.

### <a href = "https://www.acmicpc.net/problem/10746">BOJ 10746</a>

## 2024.11.05.

### <a href = "https://www.acmicpc.net/problem/17517">BOJ 17517</a>

## 2024.11.06.

### <a href = "https://www.acmicpc.net/problem/16792">BOJ 16792</a>

## 2024.11.07.

### <a href = "https://www.acmicpc.net/problem/15521">BOJ 15521</a>

```cpp
#include <bits/stdc++.h>

using ll = long long;

constexpr int MAX = 100'001;

struct Node {
    ll min = 1ll << 61, lazy = 1ll << 61;

    friend Node operator+(const Node &left, const Node &right) {
        return {std::min(left.min, right.min)};
    }
} tree[4 * MAX + 4];

struct Segtree {
    void propagate(int node, int start, int end) {
        tree[node].min = std::min(tree[node].min, tree[node].lazy);
        if (start ^ end) {
            for (const auto &i: {node << 1, node << 1 | 1}) {
                tree[i].lazy = std::min(tree[i].lazy, tree[node].lazy);
            }
        }
        tree[node].lazy = 1ll << 61;
    }

    void update(int node, int start, int end, int left, int right, ll v) {
        propagate(node, start, end);
        if (right < start || end < left) return;
        if (left <= start && end <= right) {
            tree[node].lazy = v;
            propagate(node, start, end);
            return;
        }
        int mid = (start + end) >> 1;
        update(node << 1, start, mid, left, right, v), update(node << 1 | 1, mid + 1, end, left, right, v);
        tree[node] = tree[node << 1] + tree[node << 1 | 1];
    }

    ll query(int node, int start, int end, int left, int right) {
        propagate(node, start, end);
        if (right < start || end < left) return 1ll << 61;
        if (left <= start && end <= right) return tree[node].min;
        int mid = (start + end) >> 1;
        return std::min(query(node << 1, start, mid, left, right), query(node << 1 | 1, mid + 1, end, left, right));
    }
} seg;

struct HLD {
    std::vector<int> graph[MAX + 1];
    int top[MAX + 1], in[MAX + 1], out[MAX + 1], depth[MAX + 1], sz[MAX + 1], parent[MAX + 1];

    void add(int u, int v) {
        graph[u].push_back(v);
    }

    void init(int root = 1) {
        top[root] = root;

        dfs(root);
        dfs2(root);
    }

    void dfs(int v = 1) {
        sz[v] = 1;
        for (auto &nv: graph[v]) {
            parent[nv] = v, depth[nv] = depth[v] + 1;
            dfs(nv);
            sz[v] += sz[nv];
            if (sz[nv] > sz[graph[v][0]]) std::swap(nv, graph[v][0]);
        }
    }

    void dfs2(int v = 1) {
        static int dfs_num = 0;
        in[v] = ++dfs_num;
        for (const auto &nv: graph[v]) {
            top[nv] = nv == graph[v][0] ? top[v] : nv;
            dfs2(nv);
        }
        out[v] = dfs_num;
    }

    void update(int u, int v, ll c) {
        while (top[u] != top[v]) {
            if (depth[top[u]] < depth[top[v]]) std::swap(u, v);
            int st = top[u];
            seg.update(1, 1, MAX, in[st], in[u], c);
            u = parent[st];
        }
        if (in[u] > in[v]) std::swap(u, v);
        seg.update(1, 1, MAX, in[u], in[v], c);
    }

    int lca(int u, int v) {
        for (; top[u] ^ top[v]; u = parent[top[u]])
            if (depth[top[u]] < depth[top[v]]) std::swap(u, v);
        return depth[u] < depth[v] ? u : v;
    }
} hld;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n, m, source, sink;
    std::cin >> n >> m >> source >> sink;

    std::vector<std::tuple<int, int, ll>> edge(m);
    std::vector<std::vector<std::pair<int, ll>>> graph(n + 1);
    for (auto &[u, v, c]: edge) {
        std::cin >> u >> v >> c;
        if (u > v) std::swap(u, v);
        graph[u].emplace_back(v, c), graph[v].emplace_back(u, c);
    }

    std::vector<ll> dist_root(n + 1, 1ll << 60);

    // Shortest path tree
    {
        std::priority_queue<std::pair<ll, int>> pq;
        std::vector<ll> dist(n + 1, 1ll << 60);
        pq.emplace(0, sink), dist[sink] = 0;

        std::vector<int> parent(n + 1, -1);
        while (!pq.empty()) {
            auto [d, v] = pq.top();
            pq.pop();

            if (dist[v] < -d) continue;

            for (const auto &[nv, c]: graph[v]) {
                if (dist[nv] > -d + c) {
                    parent[nv] = v;
                    dist[nv] = -d + c;
                    pq.emplace(-c + d, nv);
                }
            }
        }

        std::set<std::array<int, 2>> tree_edge;
        for (int i = 1; i <= n; i++) {
            if (i == sink) continue;
            assert(parent[i] != -1);
            hld.add(parent[i], i);
            tree_edge.insert({std::min(parent[i], i), std::max(parent[i], i)});
        }

        hld.init(sink);

        std::vector<std::vector<std::tuple<int, int, ll>>> upd(n + 1);

        for (const auto &[u, v, c]: edge) {
            if (tree_edge.count({u, v})) continue;  // shortest path tree edge
            upd[hld.lca(u, v)].emplace_back(u, v, dist[u] + c + dist[v]);
        }

        auto dfs = [&](auto &&dfs, int v) -> void {
            if (v != sink) {
                dist_root[v] = seg.query(1, 1, MAX, hld.in[v], hld.out[v]);
                if (dist_root[v] != 1ll << 60) dist_root[v] -= dist[v];
            }
            for (auto &[a, b, c]: upd[v]) hld.update(a, b, c);
            for (const auto &nv: hld.graph[v]) {
                dfs(dfs, nv);
            }
        };
        dfs(dfs, sink);
    }

    std::priority_queue<std::pair<ll, int>> pq;
    std::vector<ll> dist(n + 1, 1ll << 59);
    pq.emplace(0, sink), dist[sink] = 0;
    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();

        if (dist[v] < -d) continue;

        for (const auto &[nv, c]: graph[v]) {
            ll cc = std::max(dist_root[nv], -d + c);
            if (dist[nv] > cc) {
                dist[nv] = cc;
                pq.emplace(-cc, nv);
            }
        }
    }

    ll ans = dist[source];
    if (ans >= (1ll << 55)) ans = -1;
    std::cout << ans << "\n";
}
```

## 2024.11.08.

### <a href = "https://www.acmicpc.net/problem/19693">BOJ 19693</a>

### <a href = "https://www.acmicpc.net/problem/12736">BOJ 12736</a>

## 2024.11.09.

### <a href = "https://www.acmicpc.net/problem/13515">BOJ 13515</a>

## 2024.11.10.

### <a href = "https://www.acmicpc.net/problem/13516">BOJ 13516</a>

## 2024.11.11.

### <a href = "https://www.acmicpc.net/problem/27293">BOJ 27293</a>

$f(x) = \sum n^k$가 $k + 1$차 다항식임을 보일 수 있다. 이를 활용하면, $k + 2$개의 서로 다른 $x$에 대해 대입한 값을 알 수 있다면, 함수 $f(x)$가 유일하게 결정되고, 여기서 $f(n)$을 구하면 된다. 이는 라그랑주 보간법으로 가능하다.

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    fact[0] = 1;
    for (int i = 1; i <= 100'110; i++) fact[i] = fact[i - 1] * i;
    inv[100'110] = fact[100'110].inv();
    for (int i = 100'110; i >= 1; i--) inv[i - 1] = inv[i] * i;

    int T;
    for (std::cin >> T; T--;) {
        [&]() {
            int a, b, d;
            std::cin >> a >> b >> d;

            int n = d + 2;

            std::vector<MInt> f(n + 1);
            for (int i = 1; i <= n; i++) {
                f[i] = f[i - 1] + MInt(i).pow(d);
            }

            auto g = [&](MInt x) -> MInt {
                MInt ans = 0;
                std::vector<MInt> prefix(n + 1, 1), suffix(n + 2, 1);
                for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] * (x - i);
                for (int i = n; i >= 1; i--) suffix[i] = suffix[i + 1] * (x - i);
                for (int i = 1; i <= n; i++) {
                    if ((n - i) & 1) ans -= f[i] * prefix[i - 1] * suffix[i + 1] * inv[n - i] * inv[i - 1];
                    else ans += f[i] * prefix[i - 1] * suffix[i + 1] * inv[n - i] * inv[i - 1];
                }
                return ans;
            };

            std::cout << g(b) - g(a - 1) << "\n";
        }();
    }
}
```

## 2024.11.12.

### <a href = "https://www.acmicpc.net/problem/18539">BOJ 18539</a>

```cpp
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int k;
    std::cin >> k;

    std::vector<MInt> a(k), b(k);
    for (int x, i = 0; i < k; i++) {
        std::cin >> x;
        a[k - 1 - i] = x;
    }

    for (int x, i = 0; i < k; i++) {
        std::cin >> x;
        b[i] = x;
    }

    std::vector<T> bb(k + 1);
    for (int i = 0; i < k; i++) bb[i] = kitamasa(a, b[k - 1].v + 1 - b[i].v);
    bb[k] = kitamasa(a, b[k - 1].v + 1);

    Matrix<MInt> m(k, k + 1);
    for (int i = 0; i <= k; i++) {
        for (int j = 0; j < k; j++) {
            m[j][i] = bb[i][j];
        }
    }
    m.rref();

    for (int i = 0; i < k; i++) {
        std::cout << m[i][k] / m[i][i] << " ";
    }
}
```

## 2024.11.13.

### <a href = "https://www.acmicpc.net/problem/15527">BOJ 15527</a>

## 2024.11.14.

### <a href = "https://www.acmicpc.net/problem/19562">BOJ 19562</a>

## 2024.11.15.

### <a href = "https://www.acmicpc.net/problem/21086">BOJ 21086</a>

## 2024.11.16.

### <a href = "https://www.acmicpc.net/problem/18526">BOJ 18526</a>

이 문제에서는 문제 조건인 '어느 원도 교차하지 않는다'가 매우 중요하다. 

## 2024.11.17.

### <a href = "https://www.acmicpc.net/problem/19936">BOJ 19936</a>

### <a href = "https://www.acmicpc.net/problem/15994">BOJ 15994</a>

### <a href = "https://www.acmicpc.net/problem/21594">BOJ 21594</a>

## 2024.11.18.

### <a href = "https://www.acmicpc.net/problem/10791">BOJ 10791</a>

### <a href = "https://www.acmicpc.net/problem/25283">BOJ 25283</a>

### <a href = "https://www.acmicpc.net/problem/30523">BOJ 30523</a>

다음이 성립한다.

$$A \oplus B = A + B - 2 (A \land B), A \lor B = A + B - (A \land B)$$

그러면, 일부 $A \oplus B$를 $A \lor B$로 바꾸었을 때, 가장 큰 이득을 얻기 위해서는 $A \land B$가 큰 것 부터 선택하면 된다. 이를 위해 $A \land B$를 구해야 한다. 이는 fwht and로 간단하게 가능하다.

```cpp
#include <bits/stdc++.h>

using ll = long long;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n;
    std::cin >> n;

    ll p;
    std::cin >> p;

    auto fwht = [&](std::vector<ll> &v, bool inv = false) {
        int n = (int) v.size();

        for (int s = 2, h = 1; s <= n; s <<= 1, h <<= 1) {
            for (int l = 0; l < n; l += s) {
                for (int i = 0; i < h; i++) {
                    v[l + i] += v[l + h + i] * (inv ? -1 : 1);
                }
            }
        }
    };

    std::vector<std::array<int, 2>> v(17);
    std::vector<ll> a(1 << 17), b(1 << 17);
    for (int x, i = n; i--;) {
        std::cin >> x;
        a[x] += 1;
        for (int j = 0; j < 17; j++) v[j][(x >> j) & 1] += 1;
    }

    ll ans = 0;
    for (int x, i = n; i--;) {
        std::cin >> x;
        b[x] += 1;
        for (int j = 0; j < 17; j++) ans += (1ll << j) * v[j][(~x >> j) & 1];
    }

    fwht(a), fwht(b);
    for (int i = 0; i < 1 << 17; i++) a[i] *= b[i];
    fwht(a, true);

    for (int i = (1 << 17) - 1; i >= 0; i--) {
        auto min = std::min(a[i], p);
        ans += i * min;
        p -= min;
    }

    std::cout << ans << "\n";
}
```

### <a href = "https://www.acmicpc.net/problem/18522">BOJ 18522</a>

각 슬리퍼의 방향을 모듈로 $4$에서 생각해보면, 한 쌍의 슬리퍼를 규칙에 따라 회전시킨 이후에도 총 합이 유지됨을 알 수 있다. 그리고, 적절히 잘 하면 하나의 슬리퍼를 제외하고는 모두 원하는 방향을 맞출 수 있다. 나머지 하나의 슬리퍼는 합에 의해서 자동으로 결정된다.

그러면, 각 칸 $(x, y)$를 $(x + y)$의 기우성에 따라 나누어주면 이는 이분 그래프를 형성한다. 일단 방향을 무시하고 슬리퍼를 연결하자. 그러면 여기서 최대 매칭을 찾아주면 방향을 무시한 상황에서 최대로 매칭한 슬리퍼의 개수가 된다. 하지만 방향 조건에 의해서 마지막 하나의 슬리퍼는 방향이 맞지 않을 수 있다. 

만약, 이 매칭이 완전 매칭이 아닌 경우에는 추가적인 처리 필요 없이 최대 매칭이 답이 된다. 따라서, 완전 매칭인 경우만 마지막 하나의 슬리퍼의 방향이 맞는지를 검사하고, 방향이 맞지 않는다면 답에서 $1$을 빼주면 된다.

```cpp
#include <bits/stdc++.h>

using ll = long long;

struct Hopcroft {
    int n;

    Hopcroft(int n) : n(n) {}

    int lvl[MAX + 1], A[MAX + 1], B[MAX + 1];
    bool used[MAX + 1];

    std::set<int> graph[MAX + 1];

    void bfs() {
        std::queue<int> q;
        for (int i = 0; i < n; i++) {
            if (!used[i]) {
                q.push(i);
                lvl[i] = 0;
            } else lvl[i] = INF;
        }
        while (!q.empty()) {
            auto v = q.front();
            q.pop();

            for (const auto &nv: graph[v]) {
                if (B[nv] != -1 && lvl[B[nv]] == INF) {
                    lvl[B[nv]] = lvl[v] + 1;
                    q.push(B[nv]);
                }
            }
        }
    }

    bool dfs(int v) {
        for (const auto &nv: graph[v]) {
            if (B[nv] == -1 || (lvl[B[nv]] == lvl[v] + 1 && dfs(B[nv]))) {
                used[v] = true;
                A[v] = nv, B[nv] = v;
                return true;
            }
        }
        return false;
    }

    int karp() {
        int match = 0;
        std::fill(A, A + MAX, -1), std::fill(B, B + MAX, -1);
        while (true) {
            bfs();
            int flow = 0;
            for (int i = 0; i < n; i++) if (!used[i] && dfs(i)) flow += 1;
            match += flow;
            if (flow == 0) break;
        }
        return match;
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr), std::cout.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    Hopcroft match(n * m + 3);

    auto id = [&](int x, int y) {
        return (x - 1) * m + y;
    };

    auto add = [&](int x1, int y1, int x2, int y2) {
        match.graph[id(x1, y1)].insert(id(x2, y2));
    };

    int sum = 0;
    std::vector a(n + 1, std::vector<char>(m + 1));
    std::vector d(n + 1, std::vector<int>(m + 1));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            char c;
            std::cin >> a[i][j] >> c;
            d[i][j] = c == '^' ? 0 : c == '>' ? 1 : c == 'v' ? 2 : 3;
            sum = (sum + d[i][j]) % 4;
        }
    }

    for (int i = 1; i <= n; i++) {
        for (int j = i % 2 ? 1 : 2; j <= m; j += 2) {
            for (const auto &[x, y]: {std::array<int, 2>{i, j + 1}, {i, j - 1}, {i - 1, j}, {i + 1, j}}) {
                if (x < 1 || x > n || y < 1 || y > m || a[i][j] == a[x][y]) continue;
                add(i, j, x, y);
            }
        }
    }

    int ans = match.karp();

    if (ans * 2 == n * m) {
        auto inv = [&](int id) -> std::array<int, 2> {
            return {(id - 1) / m + 1, (id - 1) % m + 1};
        };

        for (int i = 1; i <= n; i++) {
            for (int j = i % 2 ? 1 : 2; j <= m; j += 2) {
                auto [x, y] = inv(match.A[id(i, j)]);
                if (x == i - 1) {
                    sum = (sum + 2) % 4;
                } else if (x == i + 1) {
                    sum = (sum + 2) % 4;
                } else if (y == j - 1) {
                } else if (y == j + 1) {
                } else {
                    assert(false);
                }
            }
        }
        if ((sum % 4 + 4) % 4 != 0) ans -= 1;
    }

    std::cout << ans << "\n";
}
```

### <a href = "https://www.acmicpc.net/problem/24906">BOJ 24906</a>

This problem is left as an exercise for readers. 

### <a href = "https://www.acmicpc.net/problem/14508">BOJ 14508</a>

