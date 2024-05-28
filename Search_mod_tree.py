import pandas as pd
import numpy as np
import multiprocessing
import time


Global_ppm_threshold = 5


class ModTree:

    def __init__(self, dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info):

        self.dic_aa2mod = dic_aa2mod
        self.dic_mod_info = dic_mod_info
        self.dic_mod_index_info = dic_modindex_info
        self.dic_reac_group2mod = dic_reac_group2mod
        self.dic_reac_group2modindex = dic_reac_group2modindex
        self.dic_mod2index = dic_mod2index
        self.dic_index2mod = dic_index2mod
        self.iteration = 3

    def func_find_bfs(self, mass, ppm, aa, aa_position=''):

        # first iteration
        list_mod = self.dic_aa2mod[aa]
        list_result = []
        list_index = []
        if aa_position == 'N-term':
            list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
            list_n_term = self.dic_aa2mod['N-term']
            list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
            list_mod.extend(list_n_term)
        elif aa_position == 'C-term':
            list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
            list_c_term = self.dic_aa2mod['C-term']
            list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
            list_mod.extend(list_c_term)
        elif aa_position == 'Protein-N-term':
            list_n_term = self.dic_aa2mod['N-term']
            list_mod.extend(list_n_term)
        elif aa_position == 'Protein-C-term':
            list_c_term = self.dic_aa2mod['C-term']
            list_mod.extend(list_c_term)
        list_index.append(np.array([self.dic_mod2index[i[0]] for i in list_mod], dtype=np.int16))
        list_group = [i[-1] for i in list_mod]
        matrix_mass_dev = np.array([i[1] for i in list_mod], dtype=np.float32)
        matrix_mass_dev = mass - matrix_mass_dev
        flag_mass_dev = np.abs(matrix_mass_dev) <= ppm
        list_result.append(list_index[0][flag_mass_dev])
        del flag_mass_dev

        n = 1

        while n < self.iteration:
            print(n)
            # tmp iteration
            # 当前每一轮的mass列表、group列表、修饰列表大小是相同的，每一轮根据group列表计算当前轮每一个修饰对应下一轮的可能连接修饰是什么，
            next_index = np.array([(j, ver_index) for ver_index, tmp_group in enumerate(list_group) for i in tmp_group for j in self.dic_reac_group2modindex[i]], dtype=np.int32)
            print(next_index.shape)
            matrix_mass_dev = [matrix_mass_dev[i[1]]-self.dic_mod_index_info[i[0]][1]for i in next_index]
            flag_mass_dev = np.abs(matrix_mass_dev) <= ppm
            iter_result = next_index[flag_mass_dev]

            list_index.append(next_index)
            list_result.append(iter_result)

            if n < self.iteration - 1:
                list_group = [self.dic_mod_index_info[i][-1] for i in next_index[:, 0]]
                print(len(list_group))
            n += 1
            # list_result.append(list_index[0][flag_mass_dev])
            # next_index = [[[j, ver_index] for i in tmp_group for j in self.dic_reac_group2modindex[i]] for ver_index, tmp_group in enumerate(list_group)]
            # next_index_mass = [[self.dic_mod_index_info[j[0]][1] for j in i] for i in next_index if i]
            # matrix_mass_dev = [ for i in next_index_mass]
            # matrix_mass_dev = [matrix_mass_dev[i] - 1 for j in  for i in range(matrix_mass_dev.shape[0])]
            # list_group = [self.dic_mod_index_info[i[0]][-1] for i in next_index]

        return list_result

    def func_find_all(self, list_info):

        for i in list_info:
            mass, ppm, aa, aa_position = i
            self.func_find(mass,ppm,aa,aa_position)

    def func_find(self, mass, ppm, aa, aa_position='', leaf_reac_group='', flag_group='', iteration=0):

        if -1 * ppm <= mass <= ppm:
            return [[mass]]
        if iteration >= 3:
            return []

        # update
        if iteration == 0:
            list_mod = self.dic_aa2mod[aa]
            if aa_position == 'N-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                list_mod.extend(list_n_term)
            elif aa_position == 'C-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                list_mod.extend(list_c_term)
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_mod.extend(list_n_term)
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_mod.extend(list_c_term)
        else:
            list_mod = set()
            [list_mod.update(self.dic_reac_group2mod[tmp_reac_group]) for tmp_reac_group in leaf_reac_group if tmp_reac_group in self.dic_reac_group2mod]
            # for tmp_reac_group in leaf_reac_group:
            #     if tmp_reac_group in self.dic_reac_group2mod:
            #         tmp_mod = self.dic_reac_group2mod[tmp_reac_group]
            #         list_mod.update(tmp_mod)
            list_mod = [self.dic_mod_info[i] for i in list_mod]

        iteration += 1
        res = []

        if iteration == 3:
            # print(iteration, len(list_mod))
            if flag_group == 'chemical':
                list_mod = [i for i in list_mod if i[2] != 'biological']
            list_mass = mass - np.array([i[1] for i in list_mod])
            list_index = abs(list_mass) <= ppm
            list_res = [[list_mod[i][0], list_mass[i]] for i in range(len(list_index)) if list_index[i]]
            res += list_res
        else:
            # print(iteration, len(list_mod))
            if flag_group == 'chemical':
                list_mod = [i for i in list_mod if i[2] != 'biological']
            list_mass = np.float64(mass) - np.array([i[1] for i in list_mod])
            for i, tmp_mod_info in enumerate(list_mod):
                # print(iteration, tmp_mod_info[0], len(list_mod))
                sub_res = self.func_find(list_mass[i], ppm, aa, '', tmp_mod_info[-1], tmp_mod_info[2], iteration)
                res += [[tmp_mod_info[0]] + i for i in sub_res]

        return res

    def func_find_old(self, mass, ppm, aa, aa_position='', leaf_reac_group='', flag_group='', iteration=0):

        # update
        if iteration == 0:
            list_mod = self.dic_aa2mod[aa]
            if aa_position == 'N-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                list_mod.extend(list_n_term)
            elif aa_position == 'C-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                list_mod.extend(list_c_term)
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_mod.extend(list_n_term)
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_mod.extend(list_c_term)
        else:
            list_mod = set()
            for tmp_reac_group in leaf_reac_group:
                if tmp_reac_group in self.dic_reac_group2mod:
                    tmp_mod = self.dic_reac_group2mod[tmp_reac_group]
                    list_mod.update(tmp_mod)
            list_mod = [self.dic_mod_info[i] for i in list_mod]

        iteration += 1
        res = []

        if iteration >= 3:
            return []

        print(iteration, len(list_mod))
        for i, tmp_mod_info in enumerate(list_mod):
            # print(iteration, tmp_mod_info[0], len(list_mod))
            new_flag_group = tmp_mod_info[2]
            if new_flag_group == 'biological' and flag_group == 'chemical':
                continue
            if -1 * ppm <= mass-tmp_mod_info[1] <= ppm:
                sub_res = [[]]
            else:
                sub_res = self.func_find_old(mass-tmp_mod_info[1], ppm, aa, '', tmp_mod_info[-1], new_flag_group, iteration)
            res += [[tmp_mod_info[0]] + i for i in sub_res]

        return res

    def func_score(self, list_mod_result: list, aa, aa_position=''):

        for i, mod_result in enumerate(list_mod_result):

            mod_first = mod_result[0]
            score_mod_first = sum([1 + int(i[3] != 'Anywhere') for i in self.dic_aa2mod[aa] if i[0] == mod_first])
            if aa_position == 'N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                score_mod_first += [1 for i in list_n_term if i[0] == mod_first]
            elif aa_position == 'C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                score_mod_first += [1 for i in list_c_term if i[0] == mod_first]
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                score_mod_first += [1 for i in list_n_term if i[0] == mod_first]
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                score_mod_first += [1 for i in list_c_term if i[0] == mod_first]
            list_mod_result[i] = [score_mod_first - abs(float(mod_result[-1]))] + mod_result[:-1]

        list_mod_result.sort(key=lambda x:x[0], reverse=True)

        return list_mod_result

    def func_score_GCN(self, list_mod_result: list):

        net = ModGCNScore()
        for i, mod_result in enumerate(list_mod_result):
            mod_list = mod_result[:-1]
            mod_score = 0.
            i_score = net.predict(mod_list)



def func_load_mod_rule(path_modif):


    data_known_mod = pd.read_excel(path_modif, sheet_name=0)
    data_reactive_group1 = pd.read_excel(path_modif, sheet_name=1)
    data_reactive_group3 = pd.read_excel(path_modif, sheet_name=2)

    matrix_known_mod = data_known_mod.values
    matrix_reactive_group1 = data_reactive_group1.values
    matrix_reactive_group3 = data_reactive_group3.values
    list_delta_MS, list_Description, list_aa, list_classification, list_group, list_position, list_reactive_group2 = [], [], [], [], [], [], []

    dic_aa2mod = {}
    dic_mod_info = {}
    dic_modindex_info = {}
    dic_reac_group2mod = {}
    dic_reac_group2modindex = {}
    dic_mod2index = {}
    dic_index2mod = {}

    for i in range(matrix_reactive_group1.shape[0]):
        i_reactive_group1, i_description, i_delta_mass, i_class, i_group = matrix_reactive_group1[i]
        i_delta_mass = np.float32(i_delta_mass)
        i_reactive_group2 = i_reactive_group1.split(',')
        for tmp_reactive_group in i_reactive_group2:
            if tmp_reactive_group not in dic_reac_group2mod:
                dic_reac_group2mod[tmp_reactive_group] = set()

            dic_reac_group2mod[tmp_reactive_group].add(i_description)

        if i_description not in dic_mod_info:
            dic_mod_info[i_description] = [i_description, i_delta_mass, i_group, set()]


    for i in range(matrix_known_mod.shape[0]):
        i_delta_mass, i_description, i_aa, _, i_class, i_group, _, i_position, i_reactive_group2 = matrix_known_mod[i]
        i_delta_mass = np.float32(i_delta_mass)
        if i_description not in dic_mod2index:
            index = len(dic_mod2index)
            dic_mod2index[i_description] = index
            dic_index2mod[index] = i_description
        if i_reactive_group2 == i_reactive_group2:
            i_reactive_group2 = i_reactive_group2.split(',')
            i_reactive_group2 = [j for j in i_reactive_group2 if j in dic_reac_group2mod]
        else:
            i_reactive_group2 = []
        if i_aa not in dic_aa2mod:
            dic_aa2mod[i_aa] = []
        dic_aa2mod[i_aa].append([i_description, i_delta_mass, i_group, i_position, i_reactive_group2])

    for tmp_reactive_group in dic_reac_group2mod:
        dic_reac_group2modindex[tmp_reactive_group] = set()
        for j in dic_reac_group2mod[tmp_reactive_group]:
            dic_reac_group2modindex[tmp_reactive_group].add(dic_mod2index[j])

    for i in range(matrix_reactive_group3.shape[0]):
        i_description, i_reactive_group3 = matrix_reactive_group3[i]
        if i_reactive_group3 == i_reactive_group3:
            i_reactive_group3 = i_reactive_group3.split(',')
        else:
            i_reactive_group3 = []
        if i_description not in dic_mod_info:
            print('warning! {:s} not in dic_mod_info'.format(i_description))
        else:
            for tmp_reactive_group in i_reactive_group3:
                if tmp_reactive_group in dic_reac_group2mod:
                    dic_mod_info[i_description][-1].add(tmp_reactive_group)

    for i in dic_mod_info:
        dic_modindex_info[dic_mod2index[i]] = dic_mod_info[i]

    return dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info

def read_delta_mass(path: str):

    with open(path, 'rb')as f:
        lines = f.read().decode(encoding='utf-8').split('\r\n')
    list_info = []
    for line in lines[1:]:
        if line:
            line_list = line.split('\t')
            list_info.append([float(line_list[0]), line_list[1], line_list[2]])

    return list_info


if __name__ == '__main__':

    process_num = 1
    path_modif = r'data\串联修饰库规则20231028.xlsx'
    dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info = func_load_mod_rule(path_modif)
    modTree = ModTree(dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info)

    list_info = read_delta_mass(r'D:\项目\郑大_构建修饰树\data.txt')
    path_out = r'data\data_search.txt.txt'
    f_w = open(path_out, 'w')

    if process_num > 1:
        scan_list = [[] for i in range(process_num)]
        for i in range(len(list_info)):
            scan_list[i % process_num].append(list_info[i])

        raw_pool = multiprocessing.Pool(process_num)

        results = []
        for i in range(process_num):
            if len(scan_list[i]) > 0:
                results.append(raw_pool.apply_async(modTree.func_find_all, args=(
                scan_list[i])))

        raw_pool.close()
        raw_pool.join()

        results = [result.get() for result in results]
    else:
        print(list_info)
        for i in list_info:
            print(i)
            delta_mass, aa, aa_position = i
            delta_mass = float(delta_mass)
            ppm_value = Global_ppm_threshold / 1e6 * abs(delta_mass)
            time_start = time.time()
            list_mod_tree = modTree.func_find(delta_mass, ppm_value, aa, aa_position)
            print(list_mod_tree)
            list_mod_tree = modTree.func_score(list_mod_tree, aa, aa_position)
            print(list_mod_tree)
            time_end = time.time()
            print(time_end - time_start)
            print(len(list_mod_tree))
            f_w.write('\t'.join([str(j) for j in i]) + '\n')
            f_w.write('the number of result: ' + str(len(list_mod_tree)) + '\n')
            for i, i_mod_tree in enumerate(list_mod_tree):
                f_w.write('\t'.join([str(j) for j in i_mod_tree]) + '\n')
                print(i, i_mod_tree)
            # input('123')
    # while 1:
    #     # delta_mass = input('input delta mass')
    #     # aa = input('input aa position, exit to press 0 ')
    #     delta_mass = 445.2884
    #     aa = 'Leu'
    #     aa_position = 'N-term'
    #     if aa == '0':
    #         break
    #
    #     delta_mass = float(delta_mass)
    #     ppm_value = Global_ppm_threshold / 1e6 * abs(delta_mass)
    #     time_start = time.time()
    #     list_mod_tree = modTree.func_find(delta_mass, ppm_value, aa, aa_position)
    #     time_end = time.time()
    #     print(time_end - time_start)
    #     print(len(list_mod_tree))
    #     for i, i_mod_tree in enumerate(list_mod_tree):
    #         print(i, i_mod_tree)
    #     input('123')
