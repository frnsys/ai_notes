## Neural Turing Machines

A Neural Turing Machine is a neural network enhanced with external addressable memory (and a means of interfacing with it). Like a Turing machine, it can simulate any arbitrary procedure - in fact, given an input sequence and a target output sequence, it can learn a procedure to map between the two on its own, trainable via gradient descent (as the entire thing is differentiable).

The basic architecture of NTMs is that there is a controller (which is a neural network, typically an RNN, e.g. LSTM, or a standard feedforward network), read/write heads (the write "head" actually consists of two heads, an erase and an add head, but referred to as a single head), and a memory matrix $M_t \in \mathcal R^{N \times M}$.

Each row (of which there are $N$, each of size $M$) in the memory matrix is referred to as a memory "location".

Unlike a normal Turing machine, the read and write operations are "blurry" in that they interact in some way with all elements in memory (normal Turing machines address one element at a time). There is an attentional "focus" mechanism that constrains the memory interaction to a smaller portion - each head outputs a weighting vector which determines how much it interacts (i.e. reads or writes) with each location.

At time $t$, the read head emits a (normalized) weighting vector over the $N$ locations, $w_t$.

From this we get the $M$ length read vector $r_t$:

$$
r_t = \sum_i w_t(i) M_t(i)
$$

At time $t$, the write head emits a weighting vector $w_t$ (note that the write and read heads _each_ emit their own $w_t$ that is used in the context of that head) and an erase vector $e_t$ that have $M$ elements which line in the range (0,1)$.

Using these vectors, the memory vectors $M_{t-1}(i)$ (i.e. locations) from the previous time-step are updated:

$$
\tilde M_t(i) = M_{t-1}[\mathbb 1-w_t(i)e_t]
$$

Where $\mathbb 1$ is a row vector of all ones and the multiplication against the memory location is point-wise.

Thus a memory location is erased (all elements set to zero) if $w_t$ and $e_t$ are all ones, and if either is all zeros, then the memory is unchanged.

The write head also produces an $M$ length add vector $a_t$, which is added to the memory after the erase step:

$$
M_t(i) = \tilde M_t(i) + w_t(i) a_t
$$

So, how are these weight vectors $w_t$ produced for each head?

For each head, two addressing mechanisms are combined to produce its weighting vectors:

- _content-based addressing_: focus attention on locations similar to the controller's outputted values
- _location-based addressing_: conventional lookup by location

### Content-based addressing

Each head produces a length $M$ key vector $k_t$.

$k_t$ functions as a lookup key; we want to find an entry in $M_t$ most similar to $k_t$. A similarity function $K$ (e.g. cosine similarity) is applied to $k_t$ against all entries in $M_t$. The similarity value is multiplied by a "key strength" $\beta_t > 0$, which can attenuate the focus of attention. Then the resulting vector of similarities is normalized by applying softmax. The resulting weighting vector is $w_t^c$:

$$
w_t^c(i) = \frac{\exp(\beta_t K(k_t, M_t(i)))}{\sum_j \exp (\beta_t K(k_t, M_t(j)))}
$$

### Location-based addressing

The location-based addressing mechanism is used to move across memory locations iteratively (i.e. given a current location, move to this next location; this is called a _rotational shift_) and for random-access jumps.

Each head outputs a scalar _interpolation gate_ $g_t$ in the range $(0,1)$. This is used to blend the old weighting outputted by the head, $w_{t-1}$, with the new weighting from the content-based addressing system, $w_t^c$. The result is the _gated weighting_ $w_t^g$:

$$
w_t^g = g_t w_t^c + (1-g_t)w_{t-1}
$$

If the gate is zero, the content weighting is ignored and only the previous weighting is used.

(TODO not totally clear on this part) Next, the head also emits a _shift weighting_ $s_t$ which specifies a normalized distribution over the allowed integer shifts. For example, if shifts between -1 and 1 are allowed, $s_t$ has three elements describing how much the shifts of -1, 0, and 1 are performed. One way of doing this is by adding a softmax layer of the appropriate size to the controller.

Then we apply the rotation specified by $s_t$ to $w_t^g$:

$$
\tilde w_t(i) = \sum_{j=0}^{N-1} w_t^g(j) s_t(i-j)
$$

Over time, the shift weighting, if it isn't "sharp", can cause weightings to disperse over time. For example, with permitted shifts of -1, 0, 1 and $s_t = [0.1, 0.8, 0.1]$, the single point gets slightly blurred across the three points  To counter this, each head also emits a scalar $\gamma_t \geq 1$ that is used to (re)sharpen the final weighting:

$$
w_t(i) = \frac{\tilde w_t(i)^{\gamma_t}}{\sum_j \tilde w_t(j)^{\gamma_t}}
$$

Refer to the paper for example uses.

## References

- Neural Turing Machines. Alex Graves, Greg Wayne, Ivo Danihelka. 2014.