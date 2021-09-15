class MCAgentUpdateAll(acme.Actor):
  def __init__(
      self, number_of_states, number_of_actions, behaviour_policy, 
      num_offline_updates=0, step_size=0.1): 
        self._q = np.zeros((number_of_states, number_of_actions))
        self._behaviour_policy = behaviour_policy
        self._replay_buffer = []
        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = []
        self.returns = {}

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return self._behaviour_policy(self._q[observation])


  def observe_first(self, timestep):
    """The agent is being notified that environment was reset."""
        self._state = timestep.observation
        self._replay_buffer = [self._state]

  def observe(self, action, next_timestep):
    """The agent is being notified of an environment step."""
        s = self._state
        a = action
        r = next_timestep.reward
        g = next_timestep.discount
        next_s = next_timestep.observation

        # Offline Q-value update
        self._action = a
        self._next_state = next_s

        self._replay_buffer.append((s, a, r, g, next_s))

  def update(self):    
    """Agent should update its parameters."""

        for s, a, r, g, next_s in reverse(self._replay_buffer):
            if (s, a) not in returns:
                self.returns[(s, a)] = []
            self.returns[(s, a)].append(g)
        for (s, a) in returns.keys()
            self._q[s, a] = avg(self.returns[(s,a)])