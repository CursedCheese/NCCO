import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cmaes_classed import CMAES_CLASSED
from si import SI
from base_swarm import BaseSwarm
from base_individual import BaseIndividual
from collections import deque
import random  # incrementで個体を選ぶ際に使用

"""
全ての群を入れるリストを作成
各郡には全ての個体を入れるリストを作成

群用のクラスと個体のクラスを作成
"""


class NCCO(SI):
    
    def __init__(self, problem, dim, run_number, total_settings=None):
        super().__init__(problem, dim, run_number, total_settings=total_settings)
    
    def set_variables(self, si_settings=None):
        self.algorithm_name = 'NCCO'
        
        if si_settings is not None:
            for key, value in si_settings.items():
                setattr(self, key, value)  # 読み込んだ設定を登録する
        else:
            # 定数宣言
            self.TOL = 0.1  # 2つの群がマージされるかどうかの距離の閾値 default: 10**-6
            self.NSG = 6  # 各群の最大個体数 default: 6
            self.MAX_INC = 10  # 最大インクリメント数 default: 10
            self.w = 0.5
            self.c1 = 1.0
            self.c3 = 1.0
            self.c4 = 1.0
            self.CMAES_N = 14
            self.evaluation_timing = 'end_all'  # 'end_all', 'end_each', 'increment'
            self.cmaes_update_method = 'default'  # 'default', 'restrict_sigma'
            self.sigma_lower_bound = 0.0
            self.sigma_upper_bound = 4.0
            self.weights_type = 'normal'  # 'normal', 'soft'
            self.tpso_method = 'normal'  # 'normal', 'dec_pbest', 'window'
            self.TOL_to_sigma = False
            self.latest_swarm_not_merged = False
        
        self.iter_from_generated = 0  # 最後に群が生成されてから経過したイテレーション
        self.swarms = []  # 群のリスト
        if self.problem.change_frequency is None:
            self.problem.change_frequency = 10
            self.problem.problem_settings['change_frequency'] = self.problem.change_frequency
            # raise Exception('change_frequency is None')
        
        # * global変数の設定
        # swarmクラスやindividualクラスで使用するため
        global problem
        problem = self.problem
        
    def write_settings(self):
        self.settings = {}
        self.settings['TOL'] = self.TOL
        self.settings['NSG'] = self.NSG
        self.settings['MAX_INC'] = self.MAX_INC
        self.settings['w'] = self.w
        self.settings['c1'] = self.c1
        self.settings['c3'] = self.c3
        self.settings['c4'] = self.c4
        self.settings['CMAES_N'] = self.CMAES_N
        self.settings['evaluation_timing'] = self.evaluation_timing
        self.settings['cmaes_update_method'] = self.cmaes_update_method
        self.settings['sigma_lower_bound'] = self.sigma_lower_bound
        self.settings['sigma_upper_bound'] = self.sigma_upper_bound
        self.settings['weights_type'] = self.weights_type
        self.settings['tpso_method'] = self.tpso_method
        self.settings['TOL_to_sigma'] = self.TOL_to_sigma
        self.settings['latest_swarm_not_merged'] = self.latest_swarm_not_merged
    
    def initialize_individuals(self):
        if problem.num_component == 1:
            range_from_center = (self.upper_search_bound - self.lower_search_bound) / 5
            initial_position = range_from_center * np.random.rand() - range_from_center / 2
        else:
            initial_position = None
        individual = Individual(self.problem, initial_position)
        self.generate_new_swarm(individual)
        self.calculate_offline_error()
    
    def step(self):
        if self.iteration != 0:  # 最初のiterationではinitialize_individuals()でも評価しているから
            self.evalnum_in_iter = 0
        
        # マージ処理
        self.merge()
        
        # evals = evals + 中間点の評価回数
        
        self.cmaes_updates()
        
        # インクリメント処理
        self.increment()
        
        # self.cmaes_updates()
        # evals = evals + min(|AG|, MAX_INC)
        
        # 分離処理
        self.separate()
        
        # evals = evals + 1
        
        # 群の生成
        self.generate()
        
        # evals = evals + 1
        
        self.step_last_process()

    def merge(self):
        # cbestを更新したか他の群とマージした群を取得
        updated_swarms = [swarm for swarm in self.swarms if swarm.is_updated_cbest or swarm.is_merged]
        for updated_swarm in updated_swarms:
            updated_swarm.is_updated_cbest = False
            updated_swarm.is_merged = False
        if len(updated_swarms) >= 1 and len(self.swarms) >= 2:
            for swarm in updated_swarms:
                # 最も近い群を取得
                nearest_swarm = self.search_nearest_swarm(swarm)
                if nearest_swarm is None:
                    distance = problem.search_range[0, 1] - problem.search_range[0, 0]  # arbitrary distance
                else:
                    distance = np.linalg.norm(nearest_swarm.cbest.location - swarm.cbest.location)
                
                # 最近傍のcbestとの距離がTOL以下ならマージ
                # TODO* : TOLからsigmaに変える検証
                if self.TOL_to_sigma:
                    dist_standard = swarm.cmaes.sigma
                else:
                    dist_standard = self.TOL
                if distance < dist_standard:
                # *-----------------
                    # 2つの群のどちらが優れているかを判断
                    if swarm.cbest.fitness >= nearest_swarm.cbest.fitness:
                        superior_swarm = swarm
                        inferior_swarm = nearest_swarm
                    else:
                        superior_swarm = nearest_swarm
                        inferior_swarm = swarm
                    # swarmとnearest_swarmをマージ
                    self.merging(superior_swarm, inferior_swarm)
                    if inferior_swarm in updated_swarms:  # マージされて消えた群がループの群に含まれる場合，消しとく
                        updated_swarms.remove(inferior_swarm)
                    superior_swarm.is_merged = True
                    if len(superior_swarm.individuals) > self.NSG:
                        self.delete_below_NSG(superior_swarm)
                else:
                    #* TODO: 最新の群はマージしないテスト
                    if nearest_swarm is None or (self.latest_swarm_not_merged and swarm.is_latest_generated): continue
                    # * -----------------------------
                    # 中間点を生成
                    midpoint = (swarm.cbest.location + nearest_swarm.cbest.location) / 2  # 中点を作成
                    new_individual = Individual(self.problem, midpoint)
                    # 2つの群のどちらが優れているかを判断
                    if swarm.cbest.fitness >= nearest_swarm.cbest.fitness:
                        superior_swarm = swarm
                        inferior_swarm = nearest_swarm
                    else:
                        superior_swarm = nearest_swarm
                        inferior_swarm = swarm
                    # 優れている群に中間点を追加
                    superior_swarm.individuals.append(new_individual)
                    self.evaluate(superior_swarm, new_individual)
                    
                    # 中間点の評価値 > swarm.cbestの評価値なら，2つの群をマージ
                    if new_individual.fitness >= swarm.cbest.fitness or new_individual.fitness >= nearest_swarm.cbest.fitness:  
                        # swarmとnearest_swarmをマージ
                        self.merging(superior_swarm, inferior_swarm)
                        # マージされて消えた群がループの群に含まれる場合，消しとく
                        if inferior_swarm in updated_swarms:
                            updated_swarms.remove(inferior_swarm)
                        superior_swarm.is_merged = True
                    if len(superior_swarm.individuals) > self.NSG:
                        self.delete_below_NSG(superior_swarm)
        elif len(self.swarms) == 1:
            # print('swarms is only one')
            pass
    
    def increment(self):
        # インクリメント処理する群を選択
        if len(self.swarms) <= self.MAX_INC:
            selected_swarms = self.swarms
        else:
            if np.random.uniform() < 0.5:
                self.swarms.sort(key=lambda x: x.cbest.fitness, reverse=True)
                selected_swarms = self.swarms[:self.MAX_INC]
            else:
                selected_swarms = np.random.choice(self.swarms, self.MAX_INC, replace=False)
        
        # 個体を生成するか，個体を更新する
        for swarm in selected_swarms:
            if len(swarm.individuals) < self.NSG:
                # swarmのcbest付近に1個体を生成
                # 付近というのは最近傍のcbestとの距離を直径とする球の中
                nearest_swarm = self.search_nearest_swarm(swarm)
                if nearest_swarm is None:
                    distance = problem.search_range[0, 1] - problem.search_range[0, 0]  # arbitrary distance
                else:
                    distance = np.linalg.norm(nearest_swarm.cbest.location - swarm.cbest.location)
                sampling_point = self.random_point_in_sphere(swarm.cbest.location, distance/2)
                self.generate_new_individual_in_swarm(swarm, sampling_point, evaluate=True)
            else:
                # swarmのランダムな1個体をPSOで更新
                if self.evaluation_timing == 'increment' and np.random.uniform() < 0.5:
                    cmaes_individual = np.random.choice(swarm.cmaes.individuals)
                    self.evaluate(swarm, cmaes_individual)
                else:
                    # TODO: ランダムで選ぶ処理を実装
                    self.update_by_tpso(swarm)

    def separate(self):
        # 個体数がNSGの群からランダムに１つ選択
        nsg_swarms = [swarm for swarm in self.swarms if len(swarm.individuals) == self.NSG]
        if len(nsg_swarms) > 0:
            selected_swarm = np.random.choice(nsg_swarms)
        else:
            return
        
        # 選択された群の中からランダムに1個体を選択しcbestとの距離を計算
        selected_individual = np.random.choice(selected_swarm.individuals)
        distance = np.linalg.norm(selected_individual.location - selected_swarm.cbest.location)
        if distance > self.TOL:
            # 中間点に個体を生成
            midpoint = (selected_individual.location + selected_swarm.cbest.location) / 2
            new_individual = Individual(self.problem, midpoint)
            selected_swarm.individuals.append(new_individual)
            self.evaluate(selected_swarm, new_individual)
            # 中間点よりselected_individualの方が良い場合はselected_individualを独立させる
            if new_individual.fitness < selected_individual.fitness:
                self.separating(selected_swarm, selected_individual)
            else:
                self.delete_below_NSG(selected_swarm)  # 疑似コードにはないけどこれがないと群の個体数がNSG+1個体になる
                if new_individual.fitness > selected_swarm.tpso_cbest.fitness:
                    selected_swarm.tpso_cbest = new_individual
                    if new_individual.fitness > selected_swarm.cbest.fitness:
                        selected_swarm.cbest = new_individual  # new_individual(中間点)をcbestにする
    
    def generate(self):
        #* TODO: 最新の群はマージしないテスト
        if self.latest_swarm_not_merged:
            latest_generated_swarm = next((swarm for swarm in self.swarms if swarm.is_latest_generated), None)
            if latest_generated_swarm:
                self.iter_from_generated += 1
                if self.iter_from_generated >= 5:
                    latest_generated_swarm.is_latest_generated = False
                else:
                    return
        # * --------------------
        
        self.iter_from_generated = 0
        # 条件怪しい
        if np.random.uniform() < 0.5 or len(self.swarms) < 2:
            # ランダムな位置に1個体を生成
            new_individual = Individual(self.problem)
            self.generate_new_swarm(new_individual, in_generate=True)
        else:
            # ランダムに2つの群を選択
            selected_swarms = np.random.choice(self.swarms, 2, replace=False)
            # 2つの群のcbestの一様交叉点に新しい群を生成
            uniform_index = np.random.randint(0, 2, size=problem.dim)
            cross_point = np.array([selected_swarms[idx].cbest.location[i] for i, idx in enumerate(uniform_index)])
            new_individual = Individual(self.problem, cross_point)
            self.generate_new_swarm(new_individual, in_generate=True)
    
    def search_nearest_swarm(self, swarm):
        """引数の群に最も近い群を返す

        Args:
            swarm (Swarm): 目的の群
        Returns:
            Swarm: 引数の群に最も近い群
        """
        if len(self.swarms) == 0:
            raise Exception('swarms is empty')
        
        # 自分自身以外の群を取得
        other_swarms = [other_swarm for other_swarm in self.swarms if other_swarm != swarm]
        
        if len(other_swarms) == 0:
            return None
        elif len(other_swarms) == 1:
            return other_swarms[0]
        
        nearest_swarm = other_swarms[0]  # 初期化
        nearest_dist = np.linalg.norm(nearest_swarm.cbest.location - swarm.cbest.location)  # 初期化
        
        for other_swarm in other_swarms[1:]:
            dist = np.linalg.norm(other_swarm.cbest.location - swarm.cbest.location)
            if dist < nearest_dist:
                nearest_swarm = other_swarm
                nearest_dist = dist
        
        return nearest_swarm
    
    def merging(self, swarm1, swarm2):
        """swarm1とswarm2をマージする(swarm1に統合する)
        self.swarm(群のリスト)から2つを取り除いて，swarm1にswarm2の個体を全て取り込む
        最後にまとまった群のcbestを更新して，self.swarmsに追加する

        Args:
            swarm1 (Swarm): 取り込む側の群
            swarm2 (Swarm): 取り込まれる側の群
        """
        # swarmリストから取り除く
        swarm1_cbest = swarm1.cbest
        swarm2_cbest = swarm2.cbest
        if swarm1 in self.swarms:
            self.swarms.remove(swarm1)
        else:
            raise Exception('swarm1 is not in swarms')
        if swarm2 in self.swarms:
            self.swarms.remove(swarm2)
        else:
            raise Exception('swarm2 is not in swarms')
        
        # swarm1とswarm2をマージ
        while len(swarm2.individuals) > 0:
            swarm1.individuals.append(swarm2.individuals.pop())
        
        # cbestの確認・更新
        if swarm1_cbest.fitness > swarm2_cbest.fitness:
            swarm1.cbest = swarm1_cbest
        else:
            swarm1.cbest = swarm2_cbest
            swarm1.cmaes = swarm2.cmaes
            swarm1.is_updated_cbest = True
        
        self.swarms.append(swarm1)
    
    def separating(self, swarm, individual):
        """swarmからindividualを取り除いて，新しい群を生成する

        Args:
            swarm (Swarm): individualを含む群
            individual (Individual): swarmから取り除く個体
        """
        swarm.individuals.remove(individual)  # individualを取り除く
        if len(swarm.individuals) == 0:
            self.swarms.remove(swarm)  # 個体が無くなった群は消す
        
        if swarm.cbest == individual:
            swarm.update_cbest()  # cbestを更新, cbestが変わった場合はis_updated_cbestをTrueにする
        
        # individualを唯一の個体とした新しい群を生成
        self.generate_new_swarm(individual)
    
    def cmaes_updates(self):
        for swarm in self.swarms:
            # CMA-ESの評価
            if self.evaluation_timing == 'end_all':
                for individual in swarm.cmaes.individuals:
                    self.evaluate(swarm, individual)
            elif self.evaluation_timing == 'end_each':
                individual = np.random.choice(swarm.cmaes.individuals)
                self.evaluate(swarm, individual)
                # cbestと比較し，更新
                if individual.fitness > swarm.cmaes.cbest.fitness:
                    swarm.cmaes.cbest = individual
            
            # CMA-ESのパラメータ更新
            if self.cmaes_update_method == 'default':
                swarm.cmaes.update_with_tpso_classed(self.iteration, swarm.tpso_cbest)
            elif self.cmaes_update_method == 'restrict_sigma':
                swarm.cmaes.update_with_restrict_sigma(self.iteration, swarm.tpso_cbest, self.sigma_lower_bound, self.sigma_upper_bound)
            
            # CMA-ESのサンプリング
            swarm.cmaes.sample_population_ellipse_vector_classed()
    
    def delete_below_NSG(self, swarm):
        """引数の群の要素数がNSGより多い場合，fitnessの低い個体を削除する

        Args:
            swarm (Swarm): 個体を削除する群
        """
        # fitness順でソート
        swarm.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        # NSG+1位以下の個体を削除
        if len(swarm.individuals) > self.NSG:
            swarm.individuals = swarm.individuals[:self.NSG]
    
    def random_point_in_sphere(self, center, radius):
        """centerから半径radius以内の位置にランダムな点を生成

        Args:
            center (numpy.ndarray): 中心点
            radius (float): 半径
        Returns:
            numpy.ndarray: 範囲内のランダムな点
        """
        # 単位球内のランダムな方向を生成
        direction = np.random.normal(0, 1, problem.dim)
        direction /= np.linalg.norm(direction)  # 正規化

        # 0からradiusまでのランダムな距離
        distance = radius * np.random.uniform(0, 1) ** (1/problem.dim)

        # 基準点からのランダムな点を計算
        random_point = center + direction * distance
        return random_point
    
    def update_by_tpso(self, swarm):
        """選択した群のランダムな1個体をTPSOで更新
        位置と速度を更新後，pbestとcbestも更新する

        Args:
            swarm (Swarm): 更新する個体のある群
        """

        # 更新するTPSO個体を選択
        selected_individual = np.random.choice(swarm.individuals)
        # selected_individual = self.choose_increment(swarm)  # TODO: テスト
        
        selected_individual.velocity = self.w * selected_individual.velocity + self.c3 * np.random.rand(self.dim) * (swarm.tpso_cbest.location - selected_individual.location)
        # 群の最良個体には追加の項
        if selected_individual == swarm.tpso_cbest:
            selected_individual.velocity = selected_individual.velocity + self.c4 * (swarm.cbest.location - selected_individual.location)
        
        # 位置の更新
        selected_individual.location = selected_individual.location + selected_individual.velocity
        
        np.clip(selected_individual.location, self.lower_search_bound[0], self.upper_search_bound[0], out=selected_individual.location)
        
        self.evaluate(swarm, selected_individual)
    
    def generate_new_swarm(self, first_individual, in_generate=False):
        """引数のindividualをただ1つの要素とした新しい群を生成
        新しい群を作成後，individualを追加して，群をself.swarmsに追加

        Args:
            first_individual (Individual): 新しい群の唯一の個体
            in_generate (bool, optional): generateメカニズムで呼ばれたらTrue
        """
        # 群を初期化
        new_swarm = Swarm(self.problem)
        self.swarms.append(new_swarm)
        if in_generate:
            new_swarm.is_latest_generated = True

        # 群にTPSO個体を追加
        new_swarm.individuals.append(first_individual)
        new_swarm.cbest = first_individual
        new_swarm.tpso_cbest = first_individual
        
        # 群にTCMA-ESインスタンスを追加
        cmaes = CMAES_CLASSED(problem, new_swarm, first_individual.location, 5, self.CMAES_N, self.upper_search_bound, self.lower_search_bound)
        new_swarm.cmaes = cmaes
        
        # 群に個体とCMA-ESを追加した段階でOffline-Error計算
        # 入ってきた個体が既存の個体なら直前に評価してないから計算しない
        if first_individual.fitness is None:
            self.evaluate(new_swarm, first_individual)
        
        # CMA-ESの初期評価
        for individual in cmaes.individuals:
            self.evaluate(new_swarm, individual)
    
    def generate_new_individual_in_swarm(self, swarm, location=None, evaluate=True):
        """群に新しい個体を追加する

        Args:
            swarm (Swarm): 個体を追加する群
            location (numpy.ndarray, optional): 個体を初期化する位置. Defaults to None.
            evaluate (bool, optional): 個体ができたその場で評価するか. Defaults to True.
        """
        # デフォルトでは評価値を計算する
        new_individual = Individual(self.problem, location)
        swarm.individuals.append(new_individual)
        if evaluate:
            self.evaluate(swarm, new_individual)

    def choose_increment(self, swarm):
        """インクリメントに使用する個体を群から1つ選んで返す
        選ぶ際には最後に選ばれたのが昔な個体ほど選ばれやすい

        Args:
            swarm (Swarm): インクリメントする群

        Returns:
            Individual: インクリメントする個体
        """
        weights = []
        for individual in swarm.individuals:
            w = swarm.inc_num - individual.last_inc_num
            w = max(1, w)  # 1より小さくならないように
            weights.append(w)
        
        # chosen_individual = min(swarm.individuals, key=lambda x: x.last_inc_num)
        
        # TODO: fitnessが高い個体が選ばれやすくなる
        fitnesses = np.array([individual.fitness for individual in swarm.individuals])
        min_fitness = fitnesses.min()
        adjusted_fitnesses = fitnesses - min_fitness + 1e-6  # 負の値を避けるために調整
        probabilities = adjusted_fitnesses / adjusted_fitnesses.sum()
        chosen_individual = np.random.choice(swarm.individuals, p=probabilities)
        
        # if np.random.rand() < 0.5:
        #     chosen_individual = random.choices(swarm.individuals, weights=weights, k=1)[0]
        # else:
        #     chosen_individual = min(swarm.individuals, key=lambda x: x.last_inc_num)

        # 選ばれたので last_selected を更新
        chosen_individual.last_inc_num = swarm.inc_num
        
        swarm.inc_num += 1

        return chosen_individual

    def detect_environment_change(self):
        super().detect_environment_change()
    
    def calculate_offline_error(self):
        super().calculate_offline_error()
    
    def evaluate(self, swarm, individual, is_count=True):
        super().evaluate(swarm, individual, is_count)
    
    def reevaluate(self):
        """全swarm，全粒子を現環境で再評価する(評価のカウントはしない)
        環境が変化した際などに呼び出す
        """
        for swarm in self.swarms:
            for individual in swarm.individuals:
                self.evaluate(swarm, individual, is_count=self.reevaluate_count)
            for individual in swarm.cmaes.individuals:
                self.evaluate(swarm, individual, is_count=self.reevaluate_count)
    
    def draw_colormap3d(self):
        pass
    
    def print_result(self):
        super().print_result()

class Swarm(BaseSwarm):
    def __init__(self, problem):
        super().__init__(problem)
        self.cmaes = None
        self.is_updated_cbest = True  # 最初はcbestが無かった状態から更新されるからTrue
        self.is_merged = False
        self.tpso_cbest = None  # nmmso個体の最良individual
        self.inc_num = 0  # この群が何回インクリメントしたか test
        self.is_latest_generated = False  # 最も最近生成(generate)された群
    
    def update_current_best_fitness(self, individual=None):
        """現時点での最良の評価値(self.current_best_fitness)を計算する(TSOPC用???????)

        Args:
            individual (Individual, optional): 今のベストと比較する用．絶対に現環境のものであること. Defaults to None.
        """
        if individual is not None:
            if self.current_best_eval_env_num != self.problem.env_num or self.current_best_fitness < individual.fitness:
                self.current_best_fitness = individual.fitness
                self.current_best_eval_env_num = self.problem.env_num
        else:
            # 全探索する
            tpso_best_fitness = max([individual.fitness for individual in self.individuals if individual.eval_env_num == self.problem.env_num], default=None)
            cmaes_best_fitness = max([individual.fitness for individual in self.cmaes.individuals if individual.eval_env_num == self.problem.env_num], default=None)
            if cmaes_best_fitness is None or tpso_best_fitness > cmaes_best_fitness:
                self.current_best_fitness = tpso_best_fitness
            else:
                self.current_best_fitness = cmaes_best_fitness
            self.current_best_eval_env_num = self.problem.env_num
    
    def update_gbest(self, individual=None):
        # gbestの更新の必要は無い
        pass
    
    def update_cbest(self, individual=None):
        """cbestを更新する
        individualが指定された場合，その個体がcbestを更新するか判定する
        individualが指定されない場合，現在の個体群からcbestを更新する
        
        Args:
            individual (Individual, optional): 更新する個体. Defaults to None.
        """
        
        old_cbest = self.cbest
        
        if individual is not None:
            if individual.kind == 'TPSO' and (self.tpso_cbest is None or individual.fitness > self.tpso_cbest.fitness):
                self.tpso_cbest = individual
            elif individual.kind == 'TCMAES' and (self.cmaes.cbest is None or individual.fitness > self.cmaes.cbest.fitness):
                self.cmaes.cbest = individual
            if self.cbest is None or individual.fitness > self.cbest.fitness:
                self.cbest = individual
        else:
            self.tpso_cbest = max(
                (individual for individual in self.individuals if individual.eval_env_num == self.problem.env_num),
                key=lambda x: x.fitness,
                default=None
            )
            
            self.cmaes.cbest = max(
                (individual for individual in self.cmaes.individuals if individual.eval_env_num == self.problem.env_num),
                key=lambda x: x.fitness,
                default=None
            )
        if self.tpso_cbest.fitness is None:
            self.cbest = self.cmaes.cbest
        elif self.cmaes.cbest.fitness is None:
            self.cbest = self.tpso_cbest
        elif self.tpso_cbest.fitness > self.cmaes.cbest.fitness:
            self.cbest = self.tpso_cbest
        else:
            self.cbest = self.cmaes.cbest
        
        # cbestが変わっていたらTrueにする
        if self.cbest != old_cbest:
            self.is_updated_cbest = True

class Individual(BaseIndividual):
    def __init__(self, problem, location=None, evaluate=True):
        super().__init__(problem, location, evaluate)
        self.kind = 'TPSO'
        self.last_inc_num = 0  # 最後に評価されたインクリメントはいつか