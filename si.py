import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import glob, os
import csv
from config import config
from abc import ABC, abstractmethod


"""クラスSIを継承する時の注意点
【必ずオーバーライドする必要のあるメソッド】(引数は共通にする)
・set_variables
・draw_colormap2d
・draw_colormap3d
・initialize_individuals
・step  (stepの中でdrawを呼び出す)

【基本オーバーライドすべきでないメソッド】
・__init__
・set_visual
・draw
・draw_peak2d
・draw_peak3d
・draw_trajectory2d
・draw_trajectory3d
・create_gif
・print_graph
・print_result
"""

class SI(ABC):
    
    def __init__(self, problem, dim, run_number, total_settings=None):
        si_settings = None
        if total_settings is not None:
            tmp_config = total_settings['config']
            si_settings = total_settings['si_settings']
        else:
            tmp_config = config

        
        # * hyper parameter setting
        self.MAX_ITER = tmp_config['MAX_ITER']
        self.run_number = run_number
        self.seed = run_number
        self.dim = dim
        
        self.trajectory = tmp_config['trajectory']
        self.visualize = tmp_config['visualize']
        self.save_gif = tmp_config['save_gif']
        self.is_draw_path = tmp_config['is_draw_path']
        self.is_draw_evaled_order = tmp_config['is_draw_evaled_order']
        # self.start_center = config['start_center']
        self.grid_size = tmp_config['grid_size']
        self.visualize_seed = tmp_config['visualize_seed']
        self.visualize_iteration = tmp_config['visualize_iteration']
        self.gif_save_seed = tmp_config['gif_save_seed']
        self.print_graph_seed = tmp_config['print_graph_seed']
        self.save_fig = tmp_config['save_fig']
        self.save_fig_index = tmp_config['save_fig_index']
        self.env_change_treatment = tmp_config['env_change_treatment']
        self.reevaluate_count = tmp_config['reevaluate_count']
        self.change_detection = tmp_config['change_detection']
        self.peak_found_radius = tmp_config['peak_found_radius']
        self.tag = tmp_config['tag']
        
        # seed setting
        self.seed = run_number
        np.random.seed(self.seed)
        
        # set objective function
        self.problem = problem
        
        if self.problem.num_component == 1:
            self.start_center = True
        else:
            self.start_center = False
        
        # 探索範囲
        self.lower_search_bound = np.array([lb[0] for lb in self.problem.search_range])
        self.upper_search_bound = np.array([ub[1] for ub in self.problem.search_range])
        
        # そのステップの最も良い点の位置が記録される
        self.step_best_index = None
        self.step_best_location = np.zeros((1, self.dim))
        
        # カラーマップからn個の色を取得する
        n = 20
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # nがデフォルトカラーセットの数を超える場合、色を繰り返す
        self.colors = [default_colors[i % len(default_colors)] for i in range(n)]
        
        self.set_variables(si_settings)  # 各アルゴリズムで設定する変数を設定
        # self.initialize_individuals() # * 後ろに動かしてみる
        
        # offline-error計算用
        self.last_locations = None  # 1iter前の座標
        self.last_loc_fitnesses = None  # 今の環境でのlast_locationsの評価値
        
        self.swarms = []
        self.best_individual = None
        
        self.evalnum_in_iter = 0  # そのイテレーションで何回目の評価かを記録
        
        # create result array
        self.iteration = 0
        self.offline_index = 0
        self.bbc_index = 0
        self.myoffline_index = 0
        self.offline_error = []
        self.offline_error_current_mean = []
        self.offline_performance = []
        self.bbc = []
        self.cme = []
        self.relative_error = []
        self.peak_found_ratio = []
        self.peak_found_num = []
        self.active_peak_num = []
        self.optimal_values = []
        self.peaks_fitnesses = []
        self.eval_env_num = 0  # 評価に使用した環境の番号
        self.current_best_fitness = None  # 現在の環境で評価された評価値のうち最良値
        
        self.saved = False  # 画像を保存したかどうか
        
        self.initialize_individuals()
    
    def set_variables(self):
        Exception('set_variablesメソッドをオーバーライドしてね')
    
    def set_visual(self):
        # 軌跡を描画する場合
        if self.trajectory:
            self.track_length = 100  # * 軌跡の長さを設定
            plt.ion()  # インタラクティブモードON
            if self.dim == 2:
                self.fig, self.ax = plt.subplots()
                self.tracks = np.zeros((2, self.track_length, 2))
            elif self.dim == 3:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.tracks = np.zeros((2, self.track_length, 3))  # 3次元の軌跡データ
        elif (self.visualize or self.save_gif) and self.dim == 3:  # 軌跡を描画しないけど，3次元プロットをする場合
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        # creating a path to save gif 
        if self.save_gif or self.save_fig:
            self.frames_path = "./frames/"
            # get the name of running file
            # full_path = os.path.abspath(__file__)
            algo_name = self.algorithm_name
            # remove '.py'
            # file_name = file_name[:-3]
            if self.tag != '':
                self.tag += '_'
            # self.gif_path = f'giflist/{algo_name}/'
            # フレームを保存するディレクトリを作成
            if not os.path.exists(self.frames_path):
                os.makedirs(self.frames_path)
            if not os.path.exists(f'giflist/{algo_name}'):
                os.makedirs(f'giflist/{algo_name}')
        
        # create a grid of objective function
        if self.run_number == self.visualize_seed and (self.visualize or self.save_gif) and self.dim in (2,3):
            if self.dim == 2:
                x = np.linspace(self.problem.view_range[0], self.problem.view_range[1], self.grid_size)
                y = np.linspace(self.problem.view_range[0], self.problem.view_range[1], self.grid_size)
                self.x_grid, self.y_grid = np.meshgrid(x, y)
                self.full_grids = np.stack((self.x_grid, -self.y_grid), axis=-1)
                self.full_grids = self.full_grids.reshape(-1, 2)
                self.fitnesses_grid = self.problem.obj_func(self.full_grids, is_count=False)  # create fitness_grid in initial state
            else:
                x = np.linspace(self.problem.view_range[0], self.problem.view_range[1], self.grid_size)
                y = np.linspace(self.problem.view_range[0], self.problem.view_range[1], self.grid_size)
                z = np.linspace(self.problem.view_range[0], self.problem.view_range[1], self.grid_size)
                self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z, indexing='ij')
                # 3次元グリッドを3次元配列に変換
                self.full_grids = np.stack((self.x_grid.ravel(), self.y_grid.ravel(), self.z_grid.ravel()), axis=-1)
                self.fitnesses_grid = self.problem.obj_func(self.full_grids, is_count=False).reshape(self.grid_size, self.grid_size, self.grid_size)
    
    def initialize_individuals(self):
        Exception('initialize_individualsメソッドをオーバーライドしてね')
    
    @abstractmethod
    def step(self):
        Exception('stepメソッドをオーバーライドしてね')
    
    @abstractmethod
    def reevaluate(self):
        """全swarm，全粒子を現環境で再評価する(self.reevaluatec_count==Falseの場合評価のカウントはしない)
        環境が変化した際などに呼び出す
        """
        for swarm in self.swarms:
            for individual in swarm.individuals:
                self.evaluate(swarm, individual, is_count=self.reevaluate_count)
    
    def update_best_individual(self):
        for swarm in self.swarms:
            swarm.update_cbest()
        self.best_individual = max([swarm.cbest for swarm in self.swarms], key=lambda x: x.fitness)
    
    # TODO: individualインスタンスが，自分が属するswarmの情報を持てば，引数にswarm指定する必要なくない？
    @abstractmethod
    def evaluate(self, swarm, individual, is_count=True):
        """関数評価の処理をまとめて実行する
        
        与えられたindividualを評価し，環境が変化したら変化後の処理を行う
        その後offline-errorを計算してindividualのpbestを更新する
        Args:
            swarm (Swarm): individualが所属するswarm
            individual (Individual): 評価する個体
            is_count (bool, optional): 評価回数をカウントするかどうか
        """
        before_eval_env_num = self.problem.env_num  # 評価前の環境番号を保存
        individual.last_fitness = individual.fitness
        individual.fitness = self.problem.obj_func(individual.location.reshape(1, -1), is_count)[0]
        individual.eval_env_num = self.problem.env_num  # 評価した環境番号を保存
        
        if self.change_detection == 'known' and is_count and before_eval_env_num != individual.eval_env_num:  # 評価前後で環境が変わっている場合の処理
            self.env_change_process()
        
        # 個体を評価したのが何回目かを個体に記録
        self.evalnum_in_iter += 1
        individual.evaled_order = str(self.evalnum_in_iter)
        
        # pbest, gbest, cbest更新
        individual.update_pbest()
        swarm.update_gbest(individual)
        swarm.update_cbest(individual)
        swarm.update_current_best_fitness(individual)
        
        # offline-errorの計算
        self.calculate_offline_error()
        
        # # 相対誤差の計算
        self.calculate_re(self.current_best_fitness)
        
        # ピーク発見率の計算
        self.calculate_peak_ratio()
    
    def evaluate_location(self, location):
        """locationを評価し，評価値を返す
        evaluateメソッドの個体と結びついていないlocationを評価するバージョン

        Args:
            location (numpy.ndarray): 
        """
        before_eval_env_num = self.problem.env_num  # 評価前の環境番号を保存
        fitness = self.problem.obj_func(location.reshape(1, -1))[0]
        if self.change_detection == 'known' and before_eval_env_num != self.problem.env_num:  # 評価前後で環境が変わっている場合の処理
            self.env_change_process()
        
        # current_best_fitnessの更新 メソッドを使わず直接更新
        if self.current_best_fitness is None:
            self.current_best_fitness = fitness
            self.current_best_eval_env_num = self.problem.env_num
        elif before_eval_env_num != self.problem.env_num or self.current_best_fitness > fitness:
            self.current_best_fitness = fitness
            self.current_best_eval_env_num = self.problem.env_num
        
        # offline-errorの計算
        self.calculate_offline_error()
        
        # # 相対誤差の計算
        self.calculate_re(self.current_best_fitness)
        
        # ピーク発見率の計算
        self.calculate_peak_ratio()
        
        return fitness
    
    def env_change_process(self):
        """環境が変化した時に呼ぶ関数，
        環境が変化した時の処理をまとめて行う
        """
        # 評価前後で環境が変わった場合の処理
        if self.env_change_treatment == 'remain':
            pass
        else:
            if self.env_change_treatment == 'reevaluate':
                self.reevaluate()
    
    @abstractmethod
    def detect_environment_change(self):
        """swarmのcbestの評価値が変化したかどうかを検知し，環境変化の処理を行う
        change_detectionがunknownの時に呼び出される
        """
        # remainの場合は環境変化を検知しない
        if self.env_change_treatment == 'remain':
            return
        for swarm in self.swarms:
            old_cbest_fitness = swarm.cbest.fitness
            self.evaluate(swarm, swarm.cbest)
            if old_cbest_fitness != swarm.cbest.fitness:
                self.env_change_process()
                break
    
    @abstractmethod
    def calculate_offline_error(self):
        """Offline-ErrorおよびOffline-Performanceを計算する
        
        """
        # 最良解の取得
        current_best_list = [swarm.current_best_fitness for swarm in self.swarms if swarm.current_best_eval_env_num == self.problem.env_num]
        if len(current_best_list) != 0:
            self.current_best_fitness = max(current_best_list)
        elif self.current_best_fitness is None:
            raise Exception('ここまででself.current_best_fitnessが定義されていないとおかしい')
        
        # 最適値の取得
        optimal_value = self.problem.optimal_value[self.problem.env_num]
        self.peaks_fitnesses.append(self.problem.height.flatten())
        
        # 各評価値の計算
        self.offline_error.append(optimal_value - self.current_best_fitness)
        self.offline_performance.append(self.current_best_fitness)
        self.offline_error_current_mean.append(np.mean(self.offline_error[:self.offline_index+1]))
        self.optimal_values.append(optimal_value)
        
        self.offline_index += 1

        if self.problem.eval_count == self.problem.change_frequency:
            self.bbc.append(self.current_best_fitness)
            self.bbc_index += 1

    def step_last_process(self, detect_change=True):
        
        # 環境変化のタイミングがわからず，evalで変化する場合かつ，環境変化を検知する場合(元のアルゴリズムに検知機構がある場合はdetect_change=Falseとする)
        if self.change_detection == 'unknown' and self.problem.change_trigger == 'evaluation' and detect_change:
            self.detect_environment_change()
        
        # 描画
        self.draw()
        
        # CMEの計算(イテレーションごとのエラー)
        self.cme.append(self.offline_error[-1])
        
        # # # 相対誤差の計算
        # self.calculate_re(self.current_best_fitness)
        
        # # ピーク発見率の計算
        # self.calculate_peak_ratio()
        
        # 環境変化のタイミングがiterの場合
        if self.problem.change_trigger == 'iteration' and self.iteration % self.problem.change_frequency == 0:
            self.problem.update(1) # 環境の更新
            # 環境変化のタイミングがわかる場合
            if self.change_detection == 'known':
                self.env_change_process()

    def calculate_re(self, best_fitness):
        """Relative Errorを計算する
        
        環境の最小値と最適値の差と，最良解の評価値と最適解の差の比を計算する
        """
        if self.problem.threshold == -np.inf:
            raise Exception('threshold is minus infinity, so cannot calculate relative error')
        self.relative_error.append((best_fitness - self.problem.threshold) / (self.problem.optimal_value[self.problem.env_num] - self.problem.threshold))
    
    def calculate_peak_ratio(self):
        cbests = np.array([swarm.cbest.location for swarm in self.swarms])  # 形状: (m, n)
        peaks = np.squeeze(self.problem.center, axis=0)  # 形状: (k, n)

        # 距離の計算
        distances = np.linalg.norm(cbests[:, np.newaxis, :] - peaks[np.newaxis, :, :], axis=2)  # 形状: (m, k)
        
        # 高さが十分な最適解を表すboolean配列を作成
        active_peak_mask = self.problem.is_comp_active  # 形状: (k,)

        # 各ピークに対して、閾値以下の距離が存在するか確認
        found_peaks_mask = np.any(distances <= self.peak_found_radius, axis=0)  # 形状: (k,)
        total_mask = np.logical_and(found_peaks_mask, active_peak_mask)
        found_peaks = np.sum(total_mask)
        self.peak_found_num.append(found_peaks)

        active_peak_num = np.sum(self.problem.is_comp_active)
        self.active_peak_num.append(active_peak_num)
        peak_ratio = found_peaks / active_peak_num  # 高さがself.active_peak_thresholdより大きいピークが今存在しているピークと考える
        self.peak_found_ratio.append(peak_ratio)

    def get_best_locations(self):
        # best_locations = np.array([swarm.best_location.copy() for swarm in self.swarms])
        if hasattr(self, 'step_best_location'):
            best_locations = np.array([self.step_best_location.copy() for _ in range(self.M)])
        else:
            best_locations = np.array([swarm.best_particle.location.copy() for swarm in self.swarms])
        return best_locations
    
    def reflection_method(self, location):
        """反射法

        Args:
            location (_type_): 反射法を適用したい座標
        """
        under_dim = location < self.lower_search_bound
        over_dim = location > self.upper_search_bound
        location[under_dim] = self.lower_search_bound[under_dim] + (self.lower_search_bound[under_dim] - location[under_dim])
        location[over_dim] = self.upper_search_bound[over_dim] + (self.upper_search_bound[over_dim] - location[over_dim])
    
    def append_number_to_csv(self, number, file_path='QSO_environments.csv'):
        """csvファイルに数字を追加する関数(描画の時の環境ズレを修正するために
        環境が速く進む方のアルゴリズムが，どの環境で描画しているか記録する用)

        Args:
            number (): 追加する数字
            file_path (str, optional): csvファイルのパス，ファイルは事前に作っておく．
        """
        # CSVファイルを追記モードで開く（ファイルが存在しない場合は新規作成）
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([number])  # 数字を新しい行として書き込む
    
    def read_numbers_from_csv(self, file_path='QSO_environments.csv'):
        """append_number_to_csvで作成したcsvファイルの数字をまとめたリストを返す関数

        Args:
            file_path (str, optional):append_number_to_csvで作成したcsvファイルのパス
        Returns:
            list: csvファイルに書き込まれた数字のリスト
        """
        numbers = []  # 数字を格納するためのリストを初期化

        # CSVファイルを読み込みモードで開く
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # 行が空でない場合
                    numbers.append(int(row[0]))  # 行の最初の要素（数字）をリストに追加

        return numbers

    def draw(self):
        # draw colormap
        if self.visualize_iteration[0] <= self.iteration <= self.visualize_iteration[1]:
            if self.run_number == self.visualize_seed and (self.visualize or self.save_gif):
                if self.dim == 2:
                    if self.trajectory:
                        self.draw_trajectory2d()
                    else:
                        self.draw_colormap2d()
                elif self.dim ==3:
                    if self.trajectory:
                        self.draw_trajectory3d()
                    else:
                        self.draw_colormap3d()
                else:
                    Exception('描画できない次元です')

    def draw_colormap2d(self):
        for i, swarm in enumerate(self.swarms):
            for individual in swarm.individuals:
                if self.is_draw_path:
                    self.draw_path(individual, i)
                plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10])
                if self.is_draw_evaled_order:
                    plt.text(individual.location[0], individual.location[1] + 0.1, individual.evaled_order, fontsize=8, ha='center')
                individual.evaled_order = ""
                # if individual.is_seed:
                #     plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10], marker='^')  # TODO: spsoapagの分析用
                # else:
                #     plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10])
            if hasattr(swarm, 'cmaes'):
                for individual in swarm.cmaes.individuals:
                    plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10], marker='^')
                    if self.is_draw_evaled_order:
                        plt.text(individual.location[0], individual.location[1] + 0.1, individual.evaled_order, fontsize=8, ha='center')
                    individual.evaled_order = ""
                # TODO: test: スケールを表示
                # sigma_rounded = np.round(swarm.cmaes.sigma, 3)
                # plt.text(swarm.cmaes.centroid[0]+15, swarm.cmaes.centroid[1], f'{sigma_rounded}', fontsize=8, ha='center')
                
        # ピークの周りに半径3の円を描画
        peaks = np.squeeze(self.problem.center, axis=0)  # 形状を (k, n) に変換
        for i, peak in enumerate(peaks):
            if self.problem.is_comp_active[i]:
                circle = plt.Circle((peak[0], peak[1]), radius=self.peak_found_radius, color='red', fill=False, linestyle='--')
                plt.gca().add_patch(circle)
        
        self.draw_peak2d()
        
        self.fitnesses_grid = self.problem.obj_func(self.full_grids, is_count=False).reshape(self.grid_size, self.grid_size)
        plt.imshow(self.fitnesses_grid, extent=[self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()], alpha=0.5)
        plt.title(f'seed: {self.seed}   env: {self.problem.env_num+1}  iter: {self.iteration}  current error: {self.offline_error[self.offline_index-1]:.4f}')
        # プロットの下に文字を追加(群の数，個体の数)
        if hasattr(self.swarms[0], 'cmaes'):
            num_individuals = sum([len(swarm.individuals)+swarm.cmaes.lam for swarm in self.swarms])
        else:
            num_individuals = sum([len(swarm.individuals) for swarm in self.swarms])
        info = f'swarms: {len(self.swarms)}   individuals: {num_individuals}'
        plt.figtext(0.5, 0.02, info, wrap=True, horizontalalignment='center', fontsize=12)
        plt.colorbar()
        if self.save_gif:
            plt.savefig(self.frames_path + "frame_{:03d}.png".format(self.iteration))
        if self.visualize:
            plt.pause(0.005)
        plt.clf()
    
    def draw_colormap2dtemp(self):
        for i, swarm in enumerate(self.swarms):
            for individual in swarm.individuals:
                plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10])
            if hasattr(swarm, 'cmaes'):
                for individual in swarm.cmaes.individuals:
                    plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10], marker='^')
                
        # ピークの周りに半径3の円を描画
        # peaks = np.squeeze(self.problem.center, axis=0)  # 形状を (k, n) に変換
        # for i, peak in enumerate(peaks):
        #     if self.problem.is_comp_active[i]:
        #         circle = plt.Circle((peak[0], peak[1]), radius=self.peak_found_radius, color='red', fill=False, linestyle='--')
        #         plt.gca().add_patch(circle)
        
        self.draw_peak2d()
        
        self.fitnesses_grid = self.problem.obj_func(self.full_grids, is_count=False).reshape(self.grid_size, self.grid_size)
        plt.imshow(self.fitnesses_grid, extent=[self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()], alpha=0.5)
        plt.title(f'seed: {self.seed}   env: {self.problem.env_num+1}  iter: {self.iteration}  current error: {self.offline_error[self.offline_index-1]:.4f}')
        # プロットの下に文字を追加(群の数，個体の数)
        num_individuals = sum([len(swarm.individuals)+swarm.cmaes.lam for swarm in self.swarms])
        info = f'swarms: {len(self.swarms)}   individuals: {num_individuals}'
        plt.figtext(0.5, 0.02, info, wrap=True, horizontalalignment='center', fontsize=12)
        plt.colorbar()
        if self.save_gif:
            plt.savefig(self.frames_path + "frame_{:03d}.png".format(self.iteration))
        if self.visualize:
            plt.pause(0.005)
        plt.clf()
    
    def draw_peak2d(self):
        if self.problem.modal_type != 'other':
            best_loc = self.problem.get_best_location()
            plt.scatter(best_loc[0], best_loc[1], color='green', marker='s')
        if self.problem.modal_type == 'multi':
            centers = self.problem.center.copy()
            centers = centers.reshape(-1, 2)
            centers = np.array([point for i, point in enumerate(centers) if not np.array_equal(point, best_loc) and self.problem.is_comp_active[i]])
            if centers.shape[0] > 0:
                plt.scatter(centers[:, 0], centers[:, 1], color='green', marker='x')
                plt.Circle((centers[:, 0], centers[:, 1]), radius=self.peak_found_radius, color='red', fill=False, linestyle='--')
    
    def draw_path(self, individual, swarm_id):
        plt.scatter(individual.last_location[0], individual.last_location[1], color=self.colors[swarm_id % 10], alpha=0.3)
        plt.plot([individual.last_location[0], individual.location[0]], [individual.last_location[1], individual.location[1]], color=self.colors[swarm_id % 10], alpha=0.3)
    
    def draw_peak3d(self):
        if self.problem.modal_type != 'other':
            best_loc = self.problem.get_best_location()
            self.ax.scatter(best_loc[0], best_loc[1], best_loc[2], color='green', marker='s')
        if self.problem.modal_type == 'multi':
            centers = self.problem.center.copy()
            centers = centers.reshape(-1, 3)
            centers = np.array([point for point in centers if not np.array_equal(point, best_loc)])
            if centers.shape[0] > 0:
                self.ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='green', marker='x')
    
    def draw_colormap3d(self):
        Exception('draw_colormap3dメソッドをオーバーライドしてね')
    
    def draw_vector2d(self, origin, goal, color='r'):
        """2次元ベクトルを描画する関数

        Args:
            origin (numpy.ndarray): ベクトルの始点
            goal (numpy.ndarray): ベクトルの終点
        """
        change = goal - origin  # 変位
        change = change / np.linalg.norm(change) * 20
        
        plt.quiver(origin[0], origin[1], change[0], change[1], angles='xy', scale_units='xy', scale=1, color=color)
    
    def draw_trajectory2d_legacy(self):
        # このステップの最良点とピークを組み合わせる
        points = np.array([self.step_best_location, self.problem.get_best_location()])
        # 軌跡の更新（古いデータを削除し、新しいデータを追加）
        self.tracks = np.roll(self.tracks, -1, axis=1)
        self.tracks[:, -1, :] = points

        self.ax.clear()
        
        # 描画範囲の設定
        self.ax.set_xlim(self.lower_search_bound[0], self.upper_search_bound[0])
        self.ax.set_ylim(self.lower_search_bound[1], self.upper_search_bound[1])
        
        # 凡例用の線を追加
        legend_lines = []
        
        # 各点の軌跡を描画
        for i, (color, label) in enumerate(zip(['red', 'green'], ['best particle', 'peak'])):
            # 軌跡の描画
            for j in range(self.tracks.shape[1] - 1):
                alpha = (j + 1) / self.tracks.shape[1]
                self.ax.plot(self.tracks[i, j:j+2, 0], self.tracks[i, j:j+2, 1], color=color, alpha=alpha)
            
            self.ax.scatter(points[i, 0], points[i, 1], color=color)

            # 凡例用の線を描画（透明度を設定しない）
            line, = self.ax.plot([], [], color=color, label=label)
            legend_lines.append(line)

        # 凡例の表示
        self.ax.legend(handles=legend_lines)
        # タイトル
        plt.title(f'seed: {self.seed}   environment: {self.problem.env_num}   Offline-Error: {self.offline_error[self.offline_index-1]:.4f}')
        plt.tight_layout()
        if self.save_fig and self.problem.env_num >= self.save_fig_index and not self.saved:
            plt.savefig('image/trajectory/'+self.tag+'_traj.pdf', format='pdf')
            self.saved = True
        if self.save_gif:
            plt.savefig(self.frames_path + "frame_{:03d}.png".format(self.iteration))
        plt.pause(0.005)  # 非常に短い待機時間
    
    def draw_trajectory2d(self):
        if self.problem.modal_type != 'other':
            self.centers = self.problem.get_best_location()
            # self.centers = self.centers.reshape(-1, 2)
        if self.problem.modal_type == 'multi':
            # 全てのピーク座標を取得
            centers_tmp = self.problem.center.copy()
            centers_tmp = centers_tmp.reshape(-1, 2)
            best_idx = np.where(np.all(centers_tmp == self.centers, axis=1))[0]  # 最適解のインデックスを取得
            self.centers = centers_tmp.copy()
            
            
        elif self.problem.modal_type == 'uni':
            self.centers = self.centers.reshape(1, -1)
            best_idx = 0

        if hasattr(self, 'M') and self.M > 1:
            # 全ての群の最良解を取得
            best_locations = self.get_best_locations()
        else:
            # 最良個体の座標を取得
            best_locations = self.get_best_locations() # TODO: すぐ戻す
            # best_locations = self.locations[self.step_best_index].reshape(1, -1)
        
        if self.iteration == 0:
            self.peak_tracks = np.broadcast_to(self.centers[:, np.newaxis, :], (self.problem.num_component, self.track_length, 2))
            self.solution_tracks = np.broadcast_to(best_locations[:, np.newaxis, :], (self.M, self.track_length, 2))
        self.peak_tracks = np.roll(self.peak_tracks, -1, axis=1)  # 全てのデータを1つ手前にずらす(一番古いデータを消す)
        self.solution_tracks = np.roll(self.solution_tracks, -1, axis=1)
        self.peak_tracks[:, -1, :] = self.centers.copy()  # 新しいデータを一番後ろに追加
        self.solution_tracks[:, -1, :] = best_locations.copy()

        self.ax.clear()  # 描画のクリア
        
        # 描画範囲の設定
        self.ax.set_xlim(self.problem.view_range[0], self.problem.view_range[1])
        self.ax.set_ylim(self.problem.view_range[0], self.problem.view_range[1])
        
        for i in range(self.peak_tracks.shape[0]):  # 各ピークでループ
            for j in range(self.peak_tracks.shape[1] - 1):  # 各履歴(軌跡)でループ
                alpha = (j + 1) / self.peak_tracks.shape[1]
                self.ax.plot(self.peak_tracks[i, j:j+2, 0], self.peak_tracks[i, j:j+2, 1], color='green', alpha=alpha)
            marker = 'x'
            if i == best_idx:
                marker = 's'
            self.ax.scatter(self.centers[i, 0], self.centers[i, 1], color='green', marker=marker)
        
        for i in range(self.solution_tracks.shape[0]):
            for j in range(self.solution_tracks.shape[1] - 1):
                alpha = (j + 1) / self.solution_tracks.shape[1]
                self.ax.plot(self.solution_tracks[i, j:j+2, 0], self.solution_tracks[i, j:j+2, 1], color='red', alpha=alpha)
            self.ax.scatter(best_locations[i, 0], best_locations[i, 1], color='red', marker='o')

        # タイトル
        plt.title(f'seed: {self.seed}   environment: {self.problem.env_num+1}')
        plt.tight_layout()
        if self.save_gif:
            plt.savefig(self.frames_path + "frame_{:03d}.png".format(self.problem.env_num))
        if self.save_fig and self.problem.env_num == self.save_fig_index:
            plt.savefig('image/trajectory/'+self.tag+'traj.pdf', format='pdf')  # TODO: 変える
        plt.pause(0.005)  # 非常に短い待機時間
    
    def draw_trajectory3d(self):
        # このステップの最良点とピークを組み合わせる
        points = np.array([self.step_best_location, self.problem.get_best_location()])
        # 軌跡の更新（古いデータを削除し、新しいデータを追加）
        self.tracks = np.roll(self.tracks, -1, axis=1)
        self.tracks[:, -1, :] = points

        self.ax.clear()

        # 描画範囲の設定
        self.ax.set_xlim(self.lower_search_bound[0], self.upper_search_bound[0])
        self.ax.set_ylim(self.lower_search_bound[1], self.upper_search_bound[1])
        self.ax.set_zlim(self.lower_search_bound[2], self.upper_search_bound[2])

        # 凡例用の線を追加
        legend_lines = []

        # 各点の軌跡を描画
        for i, (color, label) in enumerate(zip(['red', 'green'], ['best particle', 'peak'])):
            # 軌跡の描画
            for j in range(self.tracks.shape[1] - 1):
                alpha = (j + 1) / self.tracks.shape[1]
                self.ax.plot(self.tracks[i, j:j+2, 0], self.tracks[i, j:j+2, 1], self.tracks[i, j:j+2, 2], color=color, alpha=alpha)

            self.ax.scatter(points[i, 0], points[i, 1], points[i, 2], color=color)

            # 凡例用の線を描画（透明度を設定しない）
            line, = self.ax.plot([], [], [], color=color, label=label)
            legend_lines.append(line)

        # 凡例の表示
        self.ax.legend(handles=legend_lines)

        # タイトル
        plt.title(f'seed: {self.seed}   iteration: {self.iteration+1}   Distance: {self.dist_result[self.iteration]:.4f}')

        if self.save_gif:
            plt.savefig(self.frames_path + "frame_{:03d}.png".format(self.iteration))
        plt.pause(0.005)  # 非常に短い待機時間
    
    def snap_shot(self, save_fig=False):
        for i, swarm in enumerate(self.swarms):
            for individual in swarm.individuals:
                # TODO: テストlast_locationを薄い色で描画し、線で繋ぐ
                self.draw_path(individual, i)
                
                if individual.is_seed:
                    plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10], marker='^')  # TODO: spsoapagの分析用
                else:
                    plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10])
            if hasattr(swarm, 'cmaes'):
                for individual in swarm.cmaes.individuals:
                    plt.scatter(individual.location[0], individual.location[1], color=self.colors[i % 10], marker='^')
                
        # ピークの周りに半径3の円を描画
        peaks = np.squeeze(self.problem.center, axis=0)  # 形状を (k, n) に変換
        for i, peak in enumerate(peaks):
            if self.problem.is_comp_active[i]:
                circle = plt.Circle((peak[0], peak[1]), radius=self.peak_found_radius, color='red', fill=False, linestyle='--')
                plt.gca().add_patch(circle)
        
        self.draw_peak2d()
        
        self.fitnesses_grid = self.problem.obj_func(self.full_grids, is_count=False).reshape(self.grid_size, self.grid_size)
        plt.imshow(self.fitnesses_grid, extent=[self.x_grid.min(), self.x_grid.max(), self.y_grid.min(), self.y_grid.max()], alpha=0.5)
        # plt.title(f'seed: {self.seed}   env: {self.problem.env_num+1}  iter: {self.iteration}  current error: {self.offline_error[self.offline_index-1]:.4f}')
        # プロットの下に文字を追加(群の数，個体の数)
        if hasattr(self.swarms[0], 'cmaes'):
            num_individuals = sum([len(swarm.individuals)+swarm.cmaes.lam for swarm in self.swarms])
        else:
            num_individuals = sum([len(swarm.individuals) for swarm in self.swarms])
        info = f'swarms: {len(self.swarms)}   individuals: {num_individuals}'
        plt.figtext(0.5, 0.02, info, wrap=True, horizontalalignment='center', fontsize=12)
        plt.colorbar()
        plt.show()
    
    def create_gif(self, timestamp):
        # 保存したフレームからGIFを作成
        print('GIF作成中')
        frames = []
        imgs = glob.glob(self.frames_path + "*.png")
        imgs.sort()
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)
        
        # problem_nameのフォルダが無ければ作成
            if not os.path.exists(f"result/{config['problem_name']}"):
                os.makedirs(f"result/{config['problem_name']}")
            # timestampのフォルダが無ければ作成
            if not os.path.exists(f"result/{config['problem_name']}/{timestamp}"):
                os.makedirs(f"result/{config['problem_name']}/{timestamp}")
        
        # Save into a GIF file that loops forever
        # frames[0].save(self.gif_path+timestamp+'.gif', format='GIF',
        #             append_images=frames[1:],
        #             save_all=True,
        #             duration=200, loop=0)
        # GIFの保存場所
        frames[0].save(f'./result/{config["problem_name"]}/{timestamp}/{timestamp}.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=200, loop=0)
        # フレームの削除
        for i in imgs:
            os.remove(i)
    
    @abstractmethod
    def print_result(self):
        print('################')
        print(f'seed: {self.seed}')
        print(f'Offline Error: {np.mean(self.offline_error):.4f}')
        print(f'Offline Performance: {np.mean(self.offline_performance):.4f}')
        print(f'CME: {np.mean(self.cme):.4f}')
        # print(f'BBC: {np.mean(self.bbc[:self.bbc_index-1]):.4f}')
        print(f'RED: {self.red:.4f}')
        print(f'Peak Found Ratio: {np.mean(self.peak_found_ratio):.4f}')
    
    def print_graph(self):
        if self.visualize or (self.save_gif and self.dim == 3):
            plt.show()
        
        # Seabornのスタイルを設定
        sns.set_theme(style="whitegrid")
        x = np.arange(0, self.offline_index)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=self.offline_error[:self.offline_index], label='current error', color='blue', alpha=0.5)
        sns.lineplot(x=x, y=self.offline_error_current_mean[:self.offline_index], label='offline error', color='red')
        plt.xlabel('evaluation number')
        plt.ylabel('error')
        plt.ylim(0, 100)
        plt.title(f'seed: {self.seed}   Offline-Error')
        plt.legend(loc='upper right', fontsize=12)
        plt.show()
        
        # peak found ratioのグラフ
        x = np.arange(0, len(self.peak_found_ratio))
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=x, y=self.peak_found_ratio, label='peak found ratio', color='blue')
        plt.xlabel('iteration')
        plt.ylabel('peak found ratio')
        plt.ylim(0, self.problem.num_component+0.5)
        plt.title(f'seed: {self.seed}   Peak Found Ratio')
        plt.legend(loc='upper right')
        plt.show()