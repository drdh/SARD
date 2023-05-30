import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class DihedralGroupElement:
    def __init__(self, name, n, k, type, x_axis_index):
        self.n = n
        self.k = k
        self.type = type
        self.x_axis_index = x_axis_index
        self.permutation = np.zeros((n, n), dtype=np.int64)
        self.map_to = np.zeros(n, dtype=np.int64)
        self.name = name

        pn = 2 * np.pi / n
        if type == 'rho':  # rotation
            for i in range(n):
                self.permutation[(k + i) % n, i] = 1
                self.map_to[i] = (k + i) % n
            self.matrix = np.array([
                [np.cos(pn * k), np.sin(pn * k)],
                [np.sin(pn * k), np.cos(pn * k)],
            ])
            self.matrix[0, 1] *= -1
            self.trans_angle = lambda phi: (pn * k + phi) % (2 * np.pi)
        elif type == 'pi':  # reflection
            for i in range(n):
                self.permutation[(k - i) % n, i] = 1
                self.map_to[i] = (k - i) % n
            k0 = k - x_axis_index
            self.matrix = np.array([
                [np.cos(pn * k0), np.sin(pn * k0)],
                [np.sin(pn * k0), np.cos(pn * k0)],
            ])
            self.matrix[1, 1] *= -1
            self.trans_angle = lambda phi: (pn * k0 - phi) % (2 * np.pi)
        else:
            raise NotImplementedError

        self.permutation_inv = np.linalg.inv(self.permutation).astype(self.permutation.dtype)


class DihedralSubgroup(nn.Module):
    def __init__(self, name, n, type, group_elements, group_elements_idx, multi_table, a=0, b=0, x_axis_index=0):
        super(DihedralSubgroup, self).__init__()
        self.n = n
        self.type = type
        self.group_elements = group_elements
        self.group_elements_idx = group_elements_idx
        self.multi_table = multi_table
        self.name = name
        self.x_axis_index = x_axis_index

        self.elements = OrderedDict()
        if type == 1:  # a=d
            for i in range(n // a):
                self.elements[f'rho_{a * i}'] = group_elements[f'rho_{a * i}']
        elif type == 2:  # a=i
            self.elements['rho_0'] = group_elements['rho_0']
            self.elements[f'pi_{a}'] = group_elements[f'pi_{a}']
        elif type == 3:  # a=k, b=l
            for i in range(n // a):
                self.elements[f'rho_{a * i}'] = group_elements[f'rho_{a * i}']
                self.elements[f'pi_{b + a * i}'] = group_elements[f'pi_{b + a * i}']
                self.elements[f'pi_{(b - a * i) % n}'] = group_elements[f'pi_{(b - a * i) % n}']
        else:
            raise NotImplementedError

        elements_indicator = []
        self.elements_idx = []
        for g_name in self.group_elements.keys():
            if g_name in self.elements:
                elements_indicator.append(1.0)
                self.elements_idx.append(group_elements_idx[g_name])
            else:
                elements_indicator.append(0.0)
        self.elements_indicator = torch.tensor([elements_indicator])
        # self.elements_indicator = np.array(elements_indicator)
        self.elements_set = set(self.elements.keys())
        self.size = len(self.elements_set)

        # adjacent matrix of subgroup
        self.adj_matrix = np.zeros((n, n), dtype=np.int64)
        self.trans_matrix_name = [[None for _ in range(n)] for _ in range(n)]
        self.trans_matrix = [[None for _ in range(n)] for _ in range(n)]
        self.trans_matrix_name_all = [[[] for _ in range(n)] for _ in range(n)]
        self.trans_matrix_all = [[[] for _ in range(n)] for _ in range(n)]
        # nodes = np.arange(self.n).reshape(-1, 1)
        nodes = np.arange(self.n)
        for g_name, g in self.elements.items():
            # next_nodes = g.permutation @ nodes
            next_nodes = g.map_to[nodes]
            for i, nn in enumerate(next_nodes.reshape(-1)):
                self.adj_matrix[i, nn] = 1
                self.trans_matrix_name[i][nn] = g_name
                self.trans_matrix[i][nn] = g.matrix
                self.trans_matrix_name_all[i][nn].append(g_name)
                self.trans_matrix_all[i][nn].append(g.matrix)

        # calculate orbits
        self.orbits = OrderedDict()
        self.orbits_trans_matrix = OrderedDict()
        self.reprentative_set_defaults = OrderedDict()
        not_chosen = np.ones(n, dtype=np.int64)
        for i in range(n):
            if np.any(not_chosen):
                orbit = self.adj_matrix[i]
                representative = orbit.argmax() + 1  # index starts from 1
                orbit_id = np.arange(1, n + 1, 1)[orbit == 1]
                self.orbits[representative] = orbit_id
                orbit_trans_matrix = np.stack([self.trans_matrix[representative - 1][o - 1] for o in orbit_id], axis=0)
                self.orbits_trans_matrix[representative] = orbit_trans_matrix

                # check whether set orientation to defaults
                succ = True
                for _ in range(100):
                    xy = np.random.randn(2, 1)
                    for ID in orbit_id:
                        matrix2ID = self.trans_matrix[representative - 1][ID - 1]
                        xy_ID = matrix2ID @ xy
                        for m in self.trans_matrix_all[representative - 1][ID - 1]:
                            xy_ID_2 = m @ xy
                            distance = ((xy_ID - xy_ID_2) ** 2).sum()
                            if distance > 1e-6:
                                succ = False
                if succ:
                    self.reprentative_set_defaults[representative] = (False, None)
                else:
                    angle = 2 * np.pi / n * (representative - 1 - x_axis_index / 2)
                    self.reprentative_set_defaults[representative] = (True, np.array([
                        np.cos(angle), np.sin(angle)]))
                not_chosen[orbit == 1] = 0


class DihedralGroup:
    def __init__(self, n, x_axis_index=0):  # 0: root->1st; 1: root->12 ; 2: root->2nd
        self.n = n

        # initialize group elements
        self.group_elements = OrderedDict()
        self.group_elements_idx = OrderedDict()
        count = 0
        for type in ['rho', 'pi']:
            for k in range(n):
                name = f'{type}_{k}'
                self.group_elements[name] = DihedralGroupElement(name, n, k, type, x_axis_index=x_axis_index)
                self.group_elements_idx[name] = count
                count += 1

        # multiplication table
        self.multi_table = OrderedDict()
        for g1_name, g1 in self.group_elements.items():
            g1_multi_g2 = OrderedDict()
            for g2_name, g2 in self.group_elements.items():
                # g1_g2_perm = g1.permutation @ g2.permutation
                g1_g2_perm = g1.map_to[g2.map_to]
                g1_g2_trans = g1.matrix @ g2.matrix
                g1_multi_g2[g2_name] = None
                for g3_name, g3 in self.group_elements.items():
                    # if np.all(g3.permutation == g1_g2_perm) and \
                    #         np.abs(g1_g2_trans - g3.matrix).sum() < 1e-6:
                    if np.all(g3.map_to == g1_g2_perm) and \
                            np.abs(g1_g2_trans - g3.matrix).sum() < 1e-6:
                        g1_multi_g2[g2_name] = g3_name
            self.multi_table[g1_name] = g1_multi_g2

        # subgroups
        self.subgroups = OrderedDict()
        for d in range(1, n + 1, 1):  # type 1
            if n % d == 0:
                name = f'H_{d}'
                self.subgroups[name] = DihedralSubgroup(name, n, 1, self.group_elements, self.group_elements_idx,
                                                        self.multi_table, a=d, x_axis_index=x_axis_index)

        for i in range(0, n, 1):  # type 2
            name = f'K_{i}'
            self.subgroups[name] = DihedralSubgroup(name, n, 2, self.group_elements, self.group_elements_idx,
                                                    self.multi_table, a=i, x_axis_index=x_axis_index)
        for k in range(1, n, 1):  # type 3
            if n % k == 0:
                for l in range(0, k, 1):
                    name = f'H_{k}_{l}'
                    self.subgroups[name] = DihedralSubgroup(name, n, 3, self.group_elements, self.group_elements_idx,
                                                            self.multi_table, a=k, b=l, x_axis_index=x_axis_index)

        count = 0
        self.subgroups_idx = OrderedDict()
        self.subgroups_element_num = OrderedDict()
        for name, sub_g in self.subgroups.items():
            self.subgroups_idx[name] = count
            count += 1
            self.subgroups_element_num[name] = sub_g.size #len(sub_g.elements)

        # subgroup structure
        self.subgroup_struct = OrderedDict({
            sub_g_name: OrderedDict({
                'sub': set(),
                'super': set(),
            }) for sub_g_name in self.subgroups.keys()
        })

        for sub_g1_name, sub_g1 in self.subgroups.items():
            for sub_g2_name, sub_g2 in self.subgroups.items():
                if sub_g1.elements_set.issubset(sub_g2.elements_set):
                    # consider all sub/super-group
                    # self.subgroup_struct[sub_g1_name]['super'].add(sub_g2_name)
                    # self.subgroup_struct[sub_g2_name]['sub'].add(sub_g1_name)

                    # only consider the closest sub/super-group
                    is_sub = True
                    for sub_g3_name, sub_g3 in self.subgroups.items():
                        if sub_g3_name not in [sub_g1_name, sub_g2_name] and \
                                sub_g1.elements_set.issubset(sub_g3.elements_set) and \
                                sub_g3.elements_set.issubset(sub_g2.elements_set):
                            is_sub = False
                    if is_sub:
                        self.subgroup_struct[sub_g1_name]['super'].add(sub_g2_name)
                        self.subgroup_struct[sub_g2_name]['sub'].add(sub_g1_name)


class LearnedGroup(nn.Module):
    def __init__(self, ns, x_axis_index=0, struct_subgroup=True, updating_subgroup=True,
                 init_subgroup_name = None, subgroup_alpha_gap = 3):
        super(LearnedGroup, self).__init__()
        # self.n = n
        self.ns = ns
        self.is_struct_subgroup = struct_subgroup
        self.Gs = OrderedDict({
            i: DihedralGroup(i, x_axis_index=x_axis_index) for i in ns
        })

        self.Gs_sub_structure = {}
        for i in ns:
            self.Gs_sub_structure[i] = {}
            for sub_i_name, sub_i in self.Gs[i].subgroups.items():
                self.Gs_sub_structure[i][sub_i_name] = {}
                for j in ns:
                    if j != i:
                        self.Gs_sub_structure[i][sub_i_name][j] = set()
                        for sub_j_name, sub_j in self.Gs[j].subgroups.items():
                            if sub_i.size != sub_j.size:
                                continue
                            is_homo = True
                            for g_i in sub_i.elements.values():
                                find_one = False
                                for g_j in sub_j.elements.values():
                                    if np.allclose(g_i.matrix, g_j.matrix):
                                        find_one = True
                                        break
                                if not find_one:
                                    is_homo = False
                                    break
                            if is_homo:
                                self.Gs_sub_structure[i][sub_i_name][j].add(sub_j_name)


        # subgroup learning
        self.cur_subgroup_name = init_subgroup_name
        self.is_updating_subgroup = updating_subgroup

        self.backup_subgroup_name = self.cur_subgroup_name


        self.subgroup_alpha_gap = subgroup_alpha_gap
        self.random_prob_init = 0.0
        self.random_prob_end = 0.0
        self.random_prob_epoch = 1 # 25
        self.random_prob = self.random_prob_init  # exploration might be meaningless, because of mismatch.

        self.value_subgroup = defaultdict(lambda : 0.0)
        self.count_subgroup = defaultdict(lambda : 1e-6)
        self.count = 1
        self.info_update_subgroup = {}

        self.subgroup_embeds = dict()
        self.num_all_elements = sum(self.ns) * 2
        self.element_indicator_slice = {}
        index = 0
        for i in self.ns:
            next_ind = index + i * 2
            self.element_indicator_slice[i] = slice(index, next_ind)
            index = next_ind
        self.get_cur_subgroup_embeds()

    def _update_random_porb(self):
        self.random_prob = max(self.random_prob_end, (
                    self.random_prob_end - self.random_prob_init) / self.random_prob_epoch * self.count + self.random_prob_init)

    def _get_parsed_names(self, inpt_name=None):
        if inpt_name is not None:
            name = inpt_name.split('>')
        else:
            name = self.cur_subgroup_name.split('>')

        sub = name[0]
        alpha = int(name[1])
        supr = name[2]
        num = int(name[3])
        sub1_size = self.Gs[num].subgroups_element_num[sub]
        sub2_size = self.Gs[num].subgroups_element_num[supr]
        sub_ratio = sub1_size / sub2_size
        name_dict = {
            'sub': sub,
            'alpha': alpha,
            'super': supr,
            'num': num,
            'alpha_ratio': (1 - sub_ratio) / self.subgroup_alpha_gap * alpha + sub_ratio,
            'close_to': sub if alpha > self.subgroup_alpha_gap / 2 else supr,
            'sub_size': sub1_size,
            'super_size': sub2_size,
        }
        return name_dict

    def get_cur_num(self):
        name = self._get_parsed_names()
        return name['num']

    def get_cur_orbits(self):
        name = self._get_parsed_names()
        cur = name['close_to']
        return self.Gs[name['num']].subgroups[cur].orbits

    def get_cur_orbits_matrix(self, representative):
        name = self._get_parsed_names()
        cur = name['close_to']
        return self.Gs[name['num']].subgroups[cur].orbits_trans_matrix[representative]

    def get_cur_reprentative_set_defaults(self, representative):
        name = self._get_parsed_names()
        cur = name['close_to']
        return self.Gs[name['num']].subgroups[cur].reprentative_set_defaults[representative]


    def _update_count(self):
        self.count_subgroup[self.cur_subgroup_name] += 1
        self.count += 1

    def store_cur_subgroup_name(self):
        self.backup_subgroup_name = self.cur_subgroup_name

    def _get_in_group_neibor_subgroup_name(self, inpt_name):
        names = self._get_parsed_names(inpt_name)
        neibor = []
        if self.subgroup_alpha_gap == 1 or not np.allclose(self.random_prob, self.random_prob_end) or self.count < 20:
            assert names['sub'] == names['super']
            s = self.Gs[names['num']].subgroup_struct[names['sub']]
            for n in s['sub'].copy().union(s['super'].copy()):
                neibor.append(f"{n}>{0}>{n}>{names['num']}")
        else:
            if names['alpha'] == 0:
                assert names['sub'] == names['super']
                s = self.Gs[names['num']].subgroup_struct[names['sub']]
                for a in s['sub'].copy():
                    if a != names['super']:
                        neibor.append(f"{a}>{1}>{names['super']}>{names['num']}")
                for a in s['super'].copy():
                    if a != names['sub']:
                        neibor.append(f"{names['sub']}>{self.subgroup_alpha_gap - 1}>{a}>{names['num']}")
            else:
                if names['alpha'] == 1:
                    neibor.append(f"{names['super']}>{0}>{names['super']}>{names['num']}")
                else:
                    neibor.append(f"{names['sub']}>{names['alpha'] - 1}>{names['super']}>{names['num']}")

                if names['alpha'] == self.subgroup_alpha_gap - 1:
                    neibor.append(f"{names['sub']}>{0}>{names['sub']}>{names['num']}")
                else:
                    neibor.append(f"{names['sub']}>{names['alpha'] + 1}>{names['super']}>{names['num']}")
        return list(set(neibor))

    def get_neibor_subgroup_name(self):
        if self.is_struct_subgroup:
            names = self._get_parsed_names()
            neibor = []
            in_group_neibor = self._get_in_group_neibor_subgroup_name(self.cur_subgroup_name)
            in_group_neibor.append(self.cur_subgroup_name)

            out_group_neibor = []
            if names['alpha'] == 0:
                for j, sub_j in self.Gs_sub_structure[names['num']][names['sub']].items():
                    for sub in sub_j:
                        n = f"{sub}>{0}>{sub}>{j}"
                        out_group_neibor.append(n)
                        out_group_neibor.extend(self._get_in_group_neibor_subgroup_name(n))

            for n in in_group_neibor:
                ns = self._get_parsed_names(n)
                if ns['alpha'] == 0:
                    for j, sub_j in self.Gs_sub_structure[ns['num']][ns['sub']].items():
                        for sub in sub_j:
                            n = f"{sub}>{0}>{sub}>{j}"
                            out_group_neibor.append(n)

            neibor.extend(in_group_neibor)
            neibor.extend(out_group_neibor)
        else:
            neibor = []
            for n, g in self.Gs.items():
                for sub in g.subgroups.keys():
                    neibor.append(f"{sub}>{0}>{sub}>{n}")

        return list(set(neibor))

    def set_cur_subgroup_name(self, name):
        self.cur_subgroup_name = name

    def _set_value_subgroup(self, name, v):
        beta = 0.2
        self.value_subgroup[name] = v * beta + self.value_subgroup[name] * (1 - beta)

    def _get_value_subgroup(self, name):
        return self.value_subgroup[name]

    def update_all_values(self, value_dict):
        for k, v in value_dict.items():
            self._set_value_subgroup(k, v)


    def restore_cur_subgroup_name(self):
        self.cur_subgroup_name = self.backup_subgroup_name

    def get_cur_subgroup_symmetrizer(self):
        names = self._get_parsed_names()
        elements_sub = self.Gs[names['num']].subgroups[names['sub']].elements
        elements_super = self.Gs[names['num']].subgroups[names['super']].elements
        size_sub = len(elements_sub)
        size_super = len(elements_super)
        alpha_ratio = names['alpha_ratio']
        symmetrizer_elements = []
        for e_n, e_v in elements_super.items():
            d = {
                'matrix': e_v.matrix,
                'permutation_inv':e_v.permutation_inv,
                'ratio': alpha_ratio / size_sub if e_n in elements_sub else (1 - alpha_ratio) / (size_super - size_sub)
            }
            symmetrizer_elements.append(d)
        return symmetrizer_elements

    def _update_cur_subgroup_embeds(self):
        embed = torch.zeros(self.num_all_elements)
        name = self._get_parsed_names()
        embed_sub = self.Gs[name['num']].subgroups[name['sub']].elements_indicator
        embed_super = self.Gs[name['num']].subgroups[name['super']].elements_indicator

        ratio_super = (1 - name['alpha_ratio']) / (name['super_size'] - name['sub_size'] + 1e-6)
        ratio_sub = name['alpha_ratio'] / name['sub_size'] - ratio_super

        embed[self.element_indicator_slice[name['num']]] = embed_sub * ratio_sub + embed_super * ratio_super
        self.subgroup_embeds[self.cur_subgroup_name] = embed

    def get_cur_subgroup_embeds(self):
        if self.cur_subgroup_name not in self.subgroup_embeds:
            self._update_cur_subgroup_embeds()
        return self.subgroup_embeds[self.cur_subgroup_name]


    def update_subgroup(self, episode_reward):
        # info
        self._update_count()
        self._update_random_porb()
        neibor_subgroup = self.get_neibor_subgroup_name()

        if self.is_updating_subgroup:
            # epsilon-greedy
            neibor_subgroup_value = np.array([self._get_value_subgroup(k) for k in neibor_subgroup])
            if np.random.uniform() < self.random_prob:
                self.cur_subgroup_name = neibor_subgroup[np.argsort(neibor_subgroup_value)[-2]]
            else:
                self.cur_subgroup_name = neibor_subgroup[neibor_subgroup_value.argmax()]

    def get_info(self):
        info = {
            'subgroup_freq': {
                k: v / self.count for k, v in self.count_subgroup.items()
            },
            'subgroup_value': {
                k: v for k,v in self.value_subgroup.items()
            },
            'random_prob': self.random_prob,
        }
        info.update(self.info_update_subgroup)
        return info


if __name__ == '__main__':

    for n in range(3, 37, 1):
        total_subgroups = 0
        for j in range(1, n + 1, 1):
            if n % j == 0:
                total_subgroups += (j + 1)

        print(f"{n}: {total_subgroups}")

    G = LearnedGroup([3,4,5,6], x_axis_index=0)
    G.get_neibor_subgroup_name()
    G.get_cur_subgroup_symmetrizer()
    print(G)

