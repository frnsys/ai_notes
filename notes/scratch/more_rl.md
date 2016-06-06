[Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/). Andrej Karpathy. May 31, 2016.

Q-learning, and thus DQN, are is outperformed by _Policy Gradients_.

The policy we learn with policy gradients is a stochastic policy (i.e. it returns a probability distribution over actions, from which we draw to decide an action). This policy is learned via a _policy network_, i.e. a neural network. The policy network can be notated $p(a|s)$, i.e. it gives a distribution over actions given a state.

When working with data that has a temporal component (e.g. frames in a video game), you may want to feed frames in two at a time to capture motion, or use _difference frames_ which are formed by subtracting the current frame from the last frame.

Basically the way policy gradients work is you get your action distribution, sample an action from there, and execute that action. When you (perhaps eventually) get a reward, you set that reward scalar as the gradient for the action you took. This sets the gradient ti discourage actions with negative reward and encourage actions with positive reward. Then we carry out backprop like we usually would.

Say you are training your agent to play a game. Each playthrough of the game is called a _policy rollout_. Say you do 100 rollouts. Your agent wins 12 games and loses 88. We take all decisions made (i.e. actions taken) in the winning games and so a _positive update_ (fill in their gradients with the positive reward, run backprop, then update parameters). Then we take all the decisions in the losing games and do a _negative update_ (fill in their gradients with the negative reward, run backprop, update parameters).

Then you do another set of rollouts, perform the updates again, and so on.

With games we typically only assign rewards to the terminal states (i.e. when the game is won or lost). More generally we may have reward $r_t$ at every time step $t$. Here we might instead use a discounted reward, i.e. $R_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$. You may want to standardize these rewards (e.g. subtract mean, divide by standard deviation) before using them for backprop.

Under the hood, this basically learns a probability distribution $p(a|s)$ so that samples from the distribution maximize the reward function $r(a,s)$.

---

Good overview of shortcomings of policy gradients (in comparison to how humans learn):

> - In practical settings we usually communicate the task in some manner (e.g. English above), but in a standard RL problem you assume an arbitrary reward function that you have to discover through environment interactions. It can be argued that if a human went into game of Pong but without knowing anything about the reward function (indeed, especially if the reward function was some static but random function), the human would have a lot of difficulty learning what to do but Policy Gradients would be indifferent, and likely work much better. Similarly, if we took the frames and permuted the pixels randomly then humans would likely fail, but our Policy Gradient solution could not even tell the difference (if it’s using a fully connected network as done here).
> - A human brings in a huge amount of prior knowledge, such as intuitive physics (the ball bounces, it’s unlikely to teleport, it’s unlikely to suddenly stop, it maintains a constant velocity, etc.), and intuitive psychology (the AI opponent “wants” to win, is likely following an obvious strategy of moving towards the ball, etc.). You also understand the concept of being “in control” of a paddle, and that it responds to your UP/DOWN key commands. In contrast, our algorithms start from scratch which is simultaneously impressive (because it works) and depressing (because we lack concrete ideas for how not to).
> - Policy Gradients are a brute force solution, where the correct actions are eventually discovered and internalized into a policy. Humans build a rich, abstract model and plan within it. In Pong, I can reason that the opponent is quite slow so it might be a good strategy to bounce the ball with high vertical velocity, which would cause the opponent to not catch it in time. However, it also feels as though we also eventually “internalize” good solutions into what feels more like a reactive muscle memory policy. For example if you’re learning a new motor task (e.g. driving a car with stick shift?) you often feel yourself thinking a lot in the beginning but eventually the task becomes automatic and mindless.
> - Policy Gradients have to actually experience a positive reward, and experience it very often in order to eventually and slowly shift the policy parameters towards repeating moves that give high rewards. With our abstract model, humans can figure out what is likely to give rewards without ever actually experiencing the rewarding or unrewarding transition. I don’t have to actually experience crashing my car into a wall a few hundred times before I slowly start avoiding to do so.

Some other advice:

> In the case of Reinforcement Learning for example, one strong baseline that should always be tried first is the [cross-entropy method (CEM)](https://en.wikipedia.org/wiki/Cross-entropy_method), a simple stochastic hill-climbing “guess and check” approach inspired loosely by evolution. And if you insist on trying out Policy Gradients for your problem make sure you pay close attention to the tricks section in papers, start simple first, and use a variation of PG called [TRPO](https://arxiv.org/abs/1502.05477), which almost always works better and more consistently than vanilla PG [in practice](http://arxiv.org/abs/1604.06778). The core idea is to avoid parameter updates that change your policy too much, as enforced by a constraint on the KL divergence between the distributions predicted by the old and the new policy on a batch of data (instead of conjugate gradients the simplest instantiation of this idea could be implemented by doing a line search and checking the KL along the way).

---

[numpy implementation](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5):

```python
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
```