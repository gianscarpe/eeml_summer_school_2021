import numpy as np
import acme

class MCAgentUpdateFirst(acme.Actor):
  def __init__(
      self, number_of_states, number_of_actions, behaviour_policy):
        self._q = np.zeros((number_of_states, number_of_actions))
        self._behaviour_policy = behaviour_policy
        self._replay_buffer = []
        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = []
        self.tot_returns = np.zeros((number_of_states, number_of_actions))
        self.n_returns = np.zeros((number_of_states, number_of_actions))
        self._done = False

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return self._behaviour_policy(self._q[observation])


  def observe_first(self, timestep):
    """The agent is being notified that environment was reset."""
    self._done = False
    self._state = timestep.observation
    #self._replay_buffer = [self._state]

  def observe(self, action, next_timestep):
    """The agent is being notified of an environment step."""
    s = self._state
    a = action
    r = next_timestep.reward
    g = next_timestep.discount
    next_s = next_timestep.observation

    # Offline Q-value update
    self._action = a
    self._state = next_s
    self._done = next_timestep.last()
    
    self._replay_buffer.append((s, a, r, g, next_s))

  def update(self):    
    """Agent should update its parameters."""

    if self._done:
        idx = len(self._replay_buffer)
        discounted_return = 0
        for s, a, r, g, _ in reversed(self._replay_buffer):
            first = True
            discounted_return = g * discounted_return + r 
            for (f_s, f_a, _, _, _) in self._replay_buffer[:idx-1]:
                if s == f_s and a == f_a:
                    first = False
                    continue
            if first:
                self.tot_returns[s, a] += discounted_return
                self.n_returns[s, a] +=1
            idx -= 1
        self._q= np.nan_to_num(self.tot_returns / self.n_returns, 0)
        self._replay_buffer = []
