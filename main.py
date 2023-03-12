# %%
import math
import os

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Constants
delete_op = 0
add_op = 1
unassigned = 0
assigned = 1

# Configurations:
arraySize = 8
setSize = 50
maxSortStages = 16
max_num_of_comps = 5000
max_modifications = 6000
inversions_threshold = 2
log_path = os.path.join('Training', 'Logs')


# Utility function
def generate_binary_strings(bit_count):
    """
    generate all an array of all possible binary numbers
    :param bit_count: size of binary numbers
    :return: list of all binary numbers of that length
    """
    binary_strings = []

    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    genbin(bit_count)

    bintegers = []
    for bword in binary_strings:
        bintegers.append(np.asarray(list(map(int, bword))))
    return bintegers


def count_inversions(arr):
    """
    count the number of sorting steps needed to sort an array - inversions
    :param arr: the array
    :return: the number of inversions
    """
    _arr = arr.copy()
    if len(_arr) <= 1:
        return 0

    mid = len(_arr) // 2
    left_arr = _arr[:mid]
    right_arr = _arr[mid:]

    count = count_inversions(left_arr) + count_inversions(right_arr)

    i = j = k = 0
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] <= right_arr[j]:
            _arr[k] = left_arr[i]
            i += 1
        else:
            _arr[k] = right_arr[j]
            j += 1
            count += len(left_arr) - i
        k += 1

    while i < len(left_arr):
        _arr[k] = left_arr[i]
        i += 1
        k += 1

    while j < len(right_arr):
        _arr[k] = right_arr[j]
        j += 1
        k += 1

    return count


# all possible binary strings
all_bins = generate_binary_strings(arraySize)

# all possible binary strings that are above a certain threshold
all_bins_for_noise = list(filter(lambda x: count_inversions(x) > inversions_threshold, all_bins))


def create_random_vector_sets(set_size, vector_len, include_vectors, noise):
    """
    create a random set of vectors
    :param set_size: the size of the set
    :param vector_len: the length of the vectors
    :param include_vectors: a list of vectors that must be included
    :param noise: how much noise (random vectors) to insert
    :return: the generated set
    """
    vec_set = np.empty([set_size, vector_len])
    if noise < 0:
        noise = 0
    if noise > set_size:
        noise %= set_size

    for i in range(len(include_vectors)):
        vec_set[i] = include_vectors[i]

    if len(include_vectors) + noise <= set_size:
        sample = np.random.default_rng().choice(len(all_bins_for_noise), size=noise, replace=False)
        for i in range(len(include_vectors), len(include_vectors) + noise):
            vec_set[i] = all_bins_for_noise[sample[i - len(include_vectors)]]

    for i in range(len(include_vectors) + noise, set_size):
        vec_set[i] = np.full(vector_len, -1)

    return vec_set


# Sorting networks functions
def do_comparators_share_input(comp1, comp2):
    """
    check if to comparators share the same index
    :param comp1: first comparator
    :param comp2: second comparator
    :return: True if they share an index, False otherwise
    """
    return any(x == y for x, y in zip(comp1, comp2)) or any(x == y for y, x in zip(comp1, comp2[::-1]))


def generate_comparator_mapping(input_array_size):
    """
    generates a mapping between a comparator descriptor index to a comparator (tuple of indices)
    :param input_array_size: the input array size of the sorting network
    :return: the mapping
    """
    mapped_index = 0
    map_ = {}
    start_second_comp = 1
    for first_comp in range(input_array_size):
        for second_comp in range(start_second_comp, input_array_size):
            map_[mapped_index] = (first_comp, second_comp)
            mapped_index += 1
        start_second_comp += 1
    return map_


comparator_mapping = generate_comparator_mapping(arraySize)
num_of_comparators = len(comparator_mapping)


def is_operation_valid(network, operation):
    """
    checks the validity of an operation of a sorting network
    :param network: the sorting network
    :param operation: the operation to be executed on the sorting network
    :return: True if valid, False if invalid
    """
    op_type = operation[0]
    stage = operation[1]
    comparator_index = operation[2]

    if op_type == delete_op:
        return True
    if op_type == add_op:
        for comp in range(num_of_comparators):
            if (comp != comparator_index) \
                    and network[stage, comp] == assigned \
                    and do_comparators_share_input(comparator_mapping[comparator_index],
                                                   comparator_mapping[comp]):
                return False

    return True


def is_stage_unassigned(network, stage):
    """
    checks if a stage in a sorting network is unassigned (has to comparators)
    :param network: the sorting network
    :param stage: the stage number
    :return: True is the stage is unassigned, False otherwise
    """
    for i in range(num_of_comparators):
        if network[stage, i] != unassigned:
            return False
    return True


def modify_network(network, operation):
    """
    modify a sorting network with an operation
    :param network: the sorting network to modify
    :param operation: the operation to execute
    :return: the network, the change in the depth of the network, the change in the number of comparators
    """
    op_type = operation[0]
    stage = operation[1]
    comparator_index = operation[2]
    change_in_comparators = 0
    change_in_depth = 0
    is_previous_stage_unassigned = is_stage_unassigned(network, stage)

    if op_type == add_op and network[stage, comparator_index] == unassigned:
        network[stage, comparator_index] = assigned
        change_in_comparators = 1

    elif op_type == delete_op and network[stage, comparator_index] != unassigned:
        network[stage, comparator_index] = unassigned
        change_in_comparators -= 1

    stage_unassigned = is_stage_unassigned(network, stage)
    if stage_unassigned and not is_previous_stage_unassigned:
        change_in_depth = -1
    elif not stage_unassigned and is_previous_stage_unassigned:
        change_in_depth = 1

    return network, change_in_depth, change_in_comparators


def sort_with_network(network, vector):
    """
    sort an array with a given sorting network
    :param network: the sorting network
    :param vector: the array to sort
    :return: the array after being sorted by the network
    """
    to_sort = vector.copy()
    for stage in range(maxSortStages):
        for comp in range(num_of_comparators):
            if network[stage, comp] == assigned:
                first_index, second_index = comparator_mapping[comp]
                if to_sort[first_index] > to_sort[second_index]:
                    to_sort[first_index], to_sort[second_index] = to_sort[second_index], to_sort[first_index]
    return to_sort


current_vectors_to_sort = []


class SortingNetworkEnv(Env):
    def __init__(self):
        self.action_space = MultiDiscrete([2, maxSortStages, num_of_comparators], dtype=np.ushort)
        self.observation_space = Dict({
            'network': Box(low=0, high=1, shape=(maxSortStages, num_of_comparators), dtype=np.short),
            'sets': Box(low=0, high=1, shape=(setSize, arraySize), dtype=np.short),
            'steps_left': Discrete(max_modifications + 1),
            'network_depth': Discrete(maxSortStages + 1),
            'num_of_comps': Discrete(max_num_of_comps + 1),
            'num_of_comps_to_beat': Discrete(max_num_of_comps + 1),
            'sorted_count': Discrete(len(all_bins) + 1),
        })
        self.state = {
            'network': np.zeros((maxSortStages, num_of_comparators)),
            'sets': current_vectors_to_sort,
            'steps_left': max_modifications,
            'network_depth': 0,
            'num_of_comps': 0,
            'num_of_comps_to_beat': max_num_of_comps,
            'sorted_count': 0
        }

        self.good_networks = 0
        self.perfect_networks = 0
        self.wasted_ops = 0
        self.smallest_size = 10000
        self.start_from_empty = True
        self.fallback_state = self.state

    def step(self, action):

        # initialize step
        vecs_to_sort = self.state['sets']
        set_size = setSize
        reward = 0
        done = False

        # if no steps are left, finish the episode
        if self.state['steps_left'] == 0:
            return self.state, reward, True, {}
        self.state['steps_left'] -= 1

        # if the operation is invalid, return negative reward and stay in same step
        if not is_operation_valid(self.state['network'], action):
            reward -= set_size / 100
            return self.state, reward, done, {}

        self.state['network'], change_in_depth, change_in_comps = modify_network(self.state['network'], action)

        # if the modified network is illegal (too many stages or comparators) end the episode with a negative reward
        if (self.state['num_of_comps'] + change_in_comps > max_num_of_comps) or \
                (self.state['num_of_comps'] + change_in_comps < 0) or \
                (self.state['network_depth'] + change_in_depth > maxSortStages) or \
                (self.state['network_depth'] + change_in_depth < 0):
            return self.state, -10000, True, {}

        # if the network is unchanged, return negative reward
        if change_in_comps == 0:
            reward = -set_size / 10
            self.wasted_ops += 1
            return self.state, reward, False, {}

        self.state['network_depth'] += change_in_depth
        self.state['num_of_comps'] += change_in_comps
        steps_passed = max_modifications - self.state['steps_left']

        new_sorted_count = 0
        redundant_vectors = 0
        for vec_num in range(setSize):

            # check if vector is redundant (some feature that we tried)
            if vecs_to_sort[vec_num][0] == -1:
                redundant_vectors += 1
            else:
                after_sort = sort_with_network(self.state['network'], vecs_to_sort[vec_num])
                if np.all(after_sort[:-1] <= after_sort[1:]):
                    new_sorted_count += 1

        # check if the network sorts the whole set
        if new_sorted_count == set_size - redundant_vectors and change_in_comps != 0:
            self.good_networks += 1
            if self.state['num_of_comps'] < self.smallest_size:
                self.smallest_size = self.state['num_of_comps']
            done = self.state['num_of_comps'] < self.state['num_of_comps_to_beat']

            # the episode started from an empty network
            if self.start_from_empty:
                reward = (set_size * 100 - set_size * self.state['num_of_comps']) * 10

            # the episode started from an existing network and improved it
            elif done:
                reward = (set_size * 200 - set_size * self.state['num_of_comps']) * 10
            else:
                reward = set_size
        else:
            reward += new_sorted_count
            done = False

            # no improvement
            if self.state['sorted_count'] == new_sorted_count:
                reward -= ((set_size - new_sorted_count) + self.state['num_of_comps']) * (
                        steps_passed / max_modifications) * 5
            elif self.state['sorted_count'] > new_sorted_count:
                reward -= ((set_size - new_sorted_count) * 2 + self.state['num_of_comps']) * (
                        steps_passed / max_modifications) * 5
            # improvement
            else:
                reward = ((set_size * 50 - set_size * self.state['num_of_comps']) * (
                        new_sorted_count / set_size)) * (steps_passed / max_modifications)

        self.state['sorted_count'] = new_sorted_count
        return self.state, reward, done, {}

    def render(self):
        pass

    def reset(self):
        if self.start_from_empty:
            self.state = {
                'network': np.zeros((maxSortStages, num_of_comparators)),
                'sets': self.state['sets'],
                'steps_left': max_modifications,
                'network_depth': 0,
                'num_of_comps': 0,
                'num_of_comps_to_beat': max_num_of_comps,
                'sorted_count': 0
            }
        else:
            # copy the fallback state if the agent is set to start from an existing network
            self.state = {
                'network': self.fallback_state['network'].copy(),
                'sets': self.fallback_state['sets'],
                'steps_left': max_modifications,
                'network_depth': self.fallback_state['network_depth'],
                'num_of_comps': self.fallback_state['num_of_comps'],
                'num_of_comps_to_beat': self.fallback_state['num_of_comps_to_beat'],
                'sorted_count': self.fallback_state['sorted_count']
            }

        return self.state


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # We only log the first environment for this metric
        self.logger.record("wasted_ops", self.training_env.envs[0].wasted_ops)
        return True


if __name__ == '__main__':
    env = SortingNetworkEnv()
    num_of_envs = 12
    envs = []
    training_sets = []
    hardest_arrays = {}


    # in order to access the environment after we run the model, we need this "hack" because Stable Baselines
    # doesn't natively support this. without this, any attempts to change the vectorized environment to
    # SubProc will fail
    def make_env_supplier():
        suppliers = []
        for i in range(num_of_envs):
            suppliers.append(lambda: envs[i])
        return suppliers


    for i in range(num_of_envs):
        s = create_random_vector_sets(setSize, arraySize, [], setSize)
        training_sets.append(s)
        env = SortingNetworkEnv()
        env.state['sets'] = s
        envs.append(env)

    vec_env = DummyVecEnv([lambda e=e: e for e in envs])
    network_gen_model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=log_path, device='cuda')

    # initialize episodic data
    score = 0
    learning_timesteps = 70000
    stop_training = False
    generated_network = np.zeros((maxSortStages, num_of_comparators))
    unsorted_vectors = []
    best_networks = []

    # experimental data for "sorting forget" problem
    succ_sorted = []
    prev_succ = []

    best_sort_score = 0
    noise_addition = 0
    perfect_networks = 0
    smallest_network = max_num_of_comps
    smallest_depth = maxSortStages

    while not stop_training:

        delta_succ = [v for v in prev_succ if v not in succ_sorted]

        print("previous iteration sorted: " + str(len(prev_succ)))
        print("current iteration iteration sorted: " + str(len(succ_sorted)))
        print("sorted vectors lost since previous iteration: " + str(len(delta_succ)))

        prev_unsorted = unsorted_vectors.copy()
        prev_succ = succ_sorted.copy()

        # Print statistics
        print("current best score: " + str(best_sort_score))
        print("perfect networks: " + str(perfect_networks))
        print("smallest perfect network - size: " + str(smallest_network))
        print("smallest perfect network - depth: " + str(smallest_depth))
        print("hardest arrays: " + str(sorted(hardest_arrays.items(), key=lambda item: item[1])))

        for i in range(num_of_envs):
            envs[i].state['sets'] = training_sets[i]

        # Train the agent
        vec_env.reset()
        network_gen_model.set_env(env=vec_env)
        network_gen_model.learn(total_timesteps=learning_timesteps,
                                reset_num_timesteps=False,
                                callback=TensorboardCallback())

        # Reset the training environments and run the agent of each environment
        state = vec_env.reset()
        for agent in range(num_of_envs):

            # Environment setup
            curr_env = envs[agent]
            curr_env.good_networks = 0
            curr_env.perfect_networks = 0
            curr_env.smallest_size = 10000

            done = False
            unsorted_vectors = []

            score = 0
            steps = 0
            while not done:
                action, _ = network_gen_model.predict(curr_env.state)
                state, reward, done, info = curr_env.step(action)
                score += reward
                steps += 1
            generated_network = curr_env.state['network']

            sort_count = 0
            unsorted_vectors = []
            succ_sorted = []

            # try to sort the set with the generated network
            for i in range(setSize):
                sorted_vec = sort_with_network(generated_network, training_sets[agent][i])
                vec_to_sort = training_sets[agent][i].tolist()
                if np.all(sorted_vec[:-1] <= sorted_vec[1:]):
                    sort_count += 1
                    succ_sorted.append(vec_to_sort)
                elif vec_to_sort not in unsorted_vectors:
                    unsorted_vectors.append(vec_to_sort)
                    if str(vec_to_sort) not in hardest_arrays:
                        hardest_arrays[str(vec_to_sort)] = 0
                    hardest_arrays[str(vec_to_sort)] += 1
            if sort_count >= best_sort_score:
                best_sort_score = sort_count

            # check if network is a perfect candidate
            if sort_count == setSize:
                curr_env.start_from_empty = False
                unsorted_vectors = []
                succ_sorted = []
                sort_count = 0

                # try to sort all binary strings with the network
                for v in all_bins:
                    sorted_vec = sort_with_network(generated_network, v)
                    if np.all(sorted_vec[:-1] <= sorted_vec[1:]):
                        sort_count += 1
                        succ_sorted.append(v.tolist())
                    else:
                        unsorted_vec = v.tolist()
                        unsorted_vectors.append(unsorted_vec)
                        if str(unsorted_vec) not in hardest_arrays:
                            hardest_arrays[str(unsorted_vec)] = 0
                        hardest_arrays[str(unsorted_vec)] += 1

                if sort_count >= best_sort_score:
                    best_sort_score = sort_count

                # check if network is perfect
                if sort_count == int(math.pow(2, arraySize)):
                    print("Found perfect network after " + str(steps))
                    print(generated_network)
                    perfect_networks += 1

                    # if the network is the smallest one yet, set the new best networks
                    # (best network is an array for a future option of displaying top networks)
                    if curr_env.state['num_of_comps'] < smallest_network:
                        smallest_network = curr_env.state['num_of_comps']
                        best_networks.clear()
                        best_networks.append(curr_env.state)
                    if curr_env.state['network_depth'] < smallest_depth:
                        smallest_depth = curr_env.state['network_depth']

                    # update the new number of comparators that agent needs to beat (find network with fewer comparators)
                    # for the current environment
                    curr_env.state['num_of_comps_to_beat'] = curr_env.state['num_of_comps']

                curr_env.fallback_state = curr_env.state

            # update the set if the network didn't sort some vectors
            if len(unsorted_vectors) > 0:
                unsorted_vectors = unsorted_vectors[:setSize]
                training_sets[agent] = create_random_vector_sets(setSize, arraySize, unsorted_vectors,
                                                                 setSize - len(unsorted_vectors))
            print('on environment ' + str(agent) + ' , reward ' + str(score) + ', sorted ' + str(sort_count))

    vec_env.close()
