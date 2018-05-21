from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import os
import pickle
import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def OutOfBounds(self, idx):
        if idx < 0 or idx >= self.length:
            return True
        else:
            return False

    def OutOfBoundsStop(self, idx):
        if idx < 0 or idx > self.length:
            return True
        else:
            return False

    def __delitem__(self, key):
        # support for slicing
        if isinstance(key, slice):
            idx = key.start
            stop = max(key.stop, 0)
            if self.OutOfBounds(idx): raise KeyError()
            if self.OutOfBoundsStop(stop): raise KeyError()
            
            final_idx = self.start + stop
            if final_idx > self.maxlen:
                del self.data[(self.start + idx) : self.maxlen]
                del self.data[0 : (final_idx % self.maxlen)]
                count = final_idx - (self.start + idx)
                self.length -= count
                [self.data.append(None) for _ in range(count)]
            else:
                del self.data[(self.start + idx) : final_idx]
                count = final_idx - (self.start + idx)
                self.length -= count
                [self.data.append(None) for _ in range(count)]

        elif isinstance(key, int):
            idx = key
            del self.data[(self.start + idx) % self.maxlen]
            self.length -= 1
            self.data.append(None)

    def __getitem__(self, key):
        # support for slicing
        if isinstance(key, slice):
            idx = key.start
            stop = max(key.stop, 0)
            if self.OutOfBounds(idx): raise KeyError()
            if self.OutOfBoundsStop(stop): raise KeyError()
            
            final_idx = self.start + stop
            if final_idx > self.maxlen:
                out = self.data[(self.start + idx) : self.maxlen] + self.data[0:(final_idx % self.maxlen)]
                return out
            else:
                return self.data[(self.start + idx) : final_idx]

        elif isinstance(key, int):
            idx = key
            if self.OutOfBounds(idx):
                raise KeyError()
            return self.data[(self.start + idx) % self.maxlen]
    
    def __setitem__(self, idx, val):
        if self.OutOfBounds(idx):
            raise KeyError()
        self.data[(self.start + idx) % self.maxlen] = val

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries=False):
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    def sample(self, batch_size, batch_idxs=None):
        raise NotImplementedError()

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)

    def get_recent_state(self, current_observation):
        # This code is slightly complicated by the fact that subsequent observations might be
        # from different episodes. We ensure that an experience never spans multiple episodes.
        # This is probably not that important in practice but it seems cleaner.
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(0, self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previously handled observation was terminal, don't add the current one.
                # Otherwise we would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])
        while len(state) < self.window_length:
            state.insert(0, zeroed_observation(state[0]))
        return state

    def get_config(self):
        config = {
            'window_length': self.window_length,
            'ignore_episode_boundaries': self.ignore_episode_boundaries,
        }
        return config

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)
        
        self.limit = limit

        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        # It is not possible to tell whether the first state in the memory is terminal, because it
        # would require access to the "terminal" flag associated to the previous state. As a result
        # we will never return this first state (only using `self.terminals[0]` to know whether the
        # second state is terminal).
        # In addition we need enough entries to fill the desired window length.
        assert self.nb_entries >= self.window_length + 2, 'not enough entries in the memory'

        if batch_idxs is None:
            # Draw random indexes such that we have enough entries before each index to fill the
            # desired window length.
            batch_idxs = sample_batch_indexes(
                self.window_length, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= self.window_length + 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2]
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(self.window_length + 1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2]
            assert self.window_length + 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                assert current_idx >= 1
                current_terminal = self.terminals[current_idx - 1]
                if current_terminal and not self.ignore_episode_boundaries:
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]

            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action=action, reward=reward,
                                          state1=state1, terminal1=terminal1))
        assert len(experiences) == batch_size
        return experiences

    def append(self, observation, action, reward, terminal, training=True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)

    @property
    def nb_entries(self):
        return len(self.observations)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config

class PersistentMemory(SequentialMemory):
    '''
    this wrapper for the sequential memory class allows the user to dump 
    the experiences into storage so that learning may continue at a later time using the same replay buffer

    This is not to be used directly, but rather indirectly through the 'saveMemoryOnInterval' callback
    '''
    def __init__(self, **kwargs):
        super(PersistentMemory, self).__init__(**kwargs)

    def dump_memory(self, fpath, num_samples=None):
        # save all experiences so training can continue another time.
        # for now assumes observations are images and saves them as 
        num_samples = num_samples or self.nb_entries
        if num_samples > self.nb_entries: num_samples = self.nb_entries

        if num_samples <= 10: # just to make sure we don't accidentally overwrite
            warnings.warn("Too few number of samples. Skipping memory dump.")
            return

        minIdx = self.nb_entries - num_samples
        
        # create directory
        if not os.path.exists(fpath):
            os.makedirs(fpath) # makes all required parent directories
        else:
            warnings.warn("Memory dump directory already exists. Overwriting old data.")

        # ---- dump number of samples ----
        file_name = os.path.join(fpath, 'num_samples')
        with open(file_name,'wb') as fileObject:
            pickle.dump(num_samples, fileObject)

        # ---- save the 3 non-image ringbuffers ---
        for attr_name in ['actions', 'rewards', 'terminals']:
            file_name = os.path.join(fpath, attr_name)
            with open(file_name,'wb') as fileObject:
                attr = getattr(self, attr_name)
                del attr[0:minIdx]
                assert(len(attr) == num_samples)
                pickle.dump(attr, fileObject)   
        # ---- save all images ----
        # iterate through ring buffer. Automatically takes care of looping around
        count = 0
        for i in range(minIdx, self.nb_entries):
            file_name = os.path.join(fpath, str(count)+'.npy')
            # this works even with multiple inputs per observation, because it's all wrapped into 1 numpy array
            img = self.observations[i] 
            np.save(file_name, img)
            count += 1

    def load_memory(self, file_path, num_samples=None):
        # ---- load number of samples ----
        file_name = os.path.join(file_path, 'num_samples')
        with open(file_name,'r') as fileObject:
            old_num_samples = pickle.load(fileObject)
            
        # --- set num_samples ---
        num_samples = num_samples or old_num_samples
        if num_samples > old_num_samples: num_samples = old_num_samples
        minIdx = old_num_samples - num_samples

        # ---- load the 3 non-image ringbuffers ---
        for attr_name in ['actions', 'rewards', 'terminals']:
            file_name = os.path.join(file_path, attr_name)
            with open(file_name, 'r') as fileObject:
                attr = pickle.load(fileObject)  
                # remove samples that were before num_samples
                del attr[0:minIdx]
                assert(len(attr) == num_samples)
                # set self.attr
                setattr(self, attr_name, attr)


        # ---- load all images ----
        for i in range(minIdx, old_num_samples): # only get freshest num_samples samples
            file_name = os.path.join(file_path, str(i)+'.npy')
            img = np.load(file_name)
            self.observations.append(img)

        assert(len(self.observations) == len(self.actions))
        assert(len(self.observations) == len(self.rewards))
        assert(len(self.observations) == len(self.terminals))

        # to ensure no episode bleed-over, manually set the last last experience's terminal state to True
        # this is how keras-rl does episode barriers -- there's always a non-terminal experience after the terminal step
        if len(self.terminals) >= 2:
            idx = len(self.terminals) - 2 # can't negative index
            self.terminals[idx] = True

        print("Memory succesfully loaded. Initial size: {}".format(len(self.observations)))

class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        self.limit = limit

        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, self.nb_entries, size=batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
        return batch_params, batch_total_rewards

    def append(self, observation, action, reward, terminal, training=True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training=training)
        if training:
            self.intermediate_rewards.append(reward)

    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []

    @property
    def nb_entries(self):
        return len(self.total_rewards)

    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config['limit'] = self.limit
        return config
