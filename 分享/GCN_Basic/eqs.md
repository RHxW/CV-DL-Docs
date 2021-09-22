# aa 

$$
\frac{\partial Q}{\partial t}=-k(Q_i-Q_{i+1})-k(Q_i-Q_{i-1}) \\
\quad =k(Q_{i+1}-Q_i)-k(Q_i-Q_{i-1}) \\
\quad =k[(Q_{i+1}-Q_i)-(Q_i-Q_{i-1})] \\
=k\frac{\partial^2Q}{\partial x^2} \\
\Longrightarrow \frac{\partial Q}{\partial t}=k\frac{\partial^2Q}{\partial x^2}=k\Delta Q
$$



aa


$$
\Delta f=\frac{\partial ^2 f}{\partial x^2} + \frac{\partial ^2 f}{\partial y^2} \\
\approx
[f(x+1,y)-f(x,y)]-[f(x,y)-f(x-1,y)] \\
+[f(x,y+1)-f(x,y)]-[f(x,y)-f(x,y-1)] \\
=f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)
$$




bb
$$
A=\begin{bmatrix}
0 \quad 1 \quad 1 \quad 1 \\
1 \quad 0 \quad 0 \quad 1 \\
1 \quad 0 \quad 0 \quad 0 \\
1 \quad 1 \quad 0 \quad 0
\end{bmatrix} \
$$
D
$$
D=\begin{bmatrix}
3 \quad 0 \quad 0 \quad 0 \\
0 \quad 2 \quad 0 \quad 0 \\
0 \quad 0 \quad 1 \quad 0 \\
0 \quad 0 \quad 0 \quad 2
\end{bmatrix}
$$


bb
$$
\frac{\partial Q_1}{\partial t}=-k(T_1-T_2)-k(T_1-T_3)-k(T_1-T_4) \\
=-k\sum\limits_j A_{1j}(T_1-T_j) \\
=-k(T_1 \sum\limits_j A_{1j}-\sum\limits_jA_{1j}T_j) \\
=-kD_1T_1+k\sum\limits_jA_{1j}T_j \\
\Longrightarrow \frac{\partial Q_i}{\partial t}=-kD_iT_i+k\sum\limits_jA_{ij}T_j
$$
bb1
$$
\begin{bmatrix}
\frac{\partial Q_1}{\partial t} \\
\frac{\partial Q_2}{\partial t} \\
... \\
\frac{\partial Q_n}{\partial t}
\end{bmatrix}  = -k
\begin{bmatrix}
D_1\times T_1 \\
D_2\times T_2 \\
... \\
D_n\times T_n
\end{bmatrix}
+kA
\begin{bmatrix}
T_1 \\
T_2 \\
... \\
T_n
\end{bmatrix}
$$
bb2
$$
\frac{\partial \mathbf{Q}}{\partial t} = -kD\mathbf{T}+kA\mathbf{T} \\
=-k(D-A)\mathbf{T} \\
=-kL\mathbf{T}
$$


L
$$
L=D-A
$$
L
$$
L^{sym}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}=I_N-D^{- \frac{1}{2}}AD^{- \frac{1}{2}}
$$
L
$$
L^{sym}_{ij}=
\begin{cases}
1, \qquad \qquad \qquad \text{if} \quad i=j \quad \text{and} \quad d(v_i)\ne 0 \\
-\frac{1}{\sqrt{d(v_i)d(v_j)}}, \qquad \text{if} \{v_i, v_j\} \in E \quad \text{and} \quad i\ne j \\
0, \qquad \qquad \qquad \text{otherwise}
\end{cases}
$$


L
$$
L^{rw}=D^{-1}L=I_N-D^{-1}A
$$
L
$$
L^{rw}_{ij}=
\begin{cases}
1, \qquad \qquad \text{if} \quad i=j \quad \text{and} \quad d(v_i)\ne 0 \\
-\frac{1}{d(v_i)}, \qquad \text{if} \{v_i, v_j\} \in E \quad \text{and} \quad i\ne j \\
0, \qquad \qquad \text{otherwise}
\end{cases}
$$
C1
$$
f*h=\mathcal{F}^{-1}\{\mathcal{F(f)}\cdot \mathcal{F(h)}\} \\
=\mathcal{F}^{-1}\{\hat{f}(\omega)\cdot\hat{h}(\omega)\}
$$
c
$$
f(t)*h(t)=\int_{-\infin}^{+\infin}f(\tau)*g(t-\tau)d\tau \\
$$






F
$$
F(\omega)=\mathcal{F}[f(t)]=\int f(t)e^{-i\omega t}dt \\
f(t)=\mathcal{F}^{-1}[F(\omega)] = \frac{1}{2\pi} \int F(\omega)e^{i\omega t}d\omega
$$
ee
$$
e^{-i\omega t}
$$
FG
$$
F(\lambda_L)=\hat{f}(\lambda_L)=\sum\limits_{i=1}^N f(i)u_L(i)
$$
L
$$
L=U\Lambda U^T
$$
GC
$$
(f*h)_G=U((U^Th)\odot (U^Tf))
$$

$$
U^Th
$$

$$
U^Tf
$$

$$
U
$$

$$
x*g=U((U^Tx)\odot (U^Tg)) \\ 
=Ug_{\theta}U^Tx
$$

