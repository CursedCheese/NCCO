import numpy as np
from scipy.stats.qmc import Sobol  # 楕円体サンプリング用ソボル列
from scipy.spatial.distance import squareform, pdist  # 楕円体サンプリング用(random)
from base_swarm import BaseSwarm
from base_individual import BaseIndividual

"""借用元
https://horomary.hatenablog.com/entry/2021/01/23/013508

上記のコードを参考にして，CMA-ESの個体をクラス化したもの

基本的にはこのファイルを実行するのではなく，どこかの処理でCMAESインスタンスを作成する．
"""

class CMAES_CLASSED(BaseSwarm):
    def __init__(self, problem, parent_swarm, centroid, sigma, lam, upper_search_bound, lower_search_bound):
        super().__init__(problem)
        # 問題を受け取る
        self.problem = problem
        self.parent_swarm = parent_swarm
        
        #: 入力次元数
        self.dim = len(centroid)

        #: 世代ごと総個体数λとエリート数μ
        self.lam = lam if lam else int(4 + 3*np.log(self.dim))
        self.mu = int(np.floor(self.lam / 2))

        #: 正規分布中心とその学習率
        self.centroid = np.array(centroid, dtype=np.float64)
        # self.old_centroid = self.centroid
        self.c_m = 1.0  # default: 1.0

        #: 順位重み係数(デフォルト)
        weights = [np.log(0.5*(self.lam + 1)) - np.log(i) for i in range(1, 1+self.mu)]
        weights = np.array(weights).reshape(1, -1)
        self.weights = weights / weights.sum()
        
        # 順位重み係数(ゆるやか)
        # weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        # weights = np.array(weights).reshape(1, -1)
        # self.weights = weights / np.sum(weights)
        
        self.mu_eff = 1. / (self.weights ** 2).sum()

        #: ステップサイズ： 進化パスpと学習率c
        self.sigma = float(sigma)
        self.p_sigma = np.zeros(self.dim)
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(
            0, np.sqrt((self.mu_eff - 1)/(self.dim + 1)) - 1
            ) + self.c_sigma

        #: 共分散行列： 進化パスpとrank-μ, rank-one更新の学習率c
        self.C = np.identity(self.dim)
        self.p_c = np.zeros(self.dim)
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2.0 / ((self.dim+1.3)**2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2.0 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff)
            )
        
        # 追加
        self.best_idx = 0  # 一番良いサンプリング点のインデックス
        self.individuals = [Individual(self.problem, location=centroid, evaluate=False) for _ in range(self.lam)]
        self.upper_search_bound = upper_search_bound
        self.lower_search_bound = lower_search_bound
        self.sample_population_ellipse_vector_classed()
        self.cbest = self.individuals[0]
        self.last_locations = None  # 1iter前のlocations
        self.fitnesses_stock = None
        self.fitnesses = np.array([-9999 for _ in range(self.lam)])
        self.eval_env_num = np.array([0 for _ in range(self.lam)])
        self.last_loc_fitnesses = None  # 時間差CMA-ESで使用？
        self.diff_fitnesses = None
        self.diag_diff_count = 0
        self.vals_num = 0  # eigvalsを平均した個数
        self.eigvals_storage = np.array([])
        self.ellipse_percent = 0.0
        self.scales = np.zeros((1, 2))
        
        # self.gbest_location = None
        # self.gbest_fitness = None
        # self.gbest_eval_env_num = None
    
    def update_gbest(self):
        """Swarmのgbestを更新する
        """
        super().update_gbest()

    
    def sample_population_ellipse_vector_classed(self):
        assert self.lam >= self.dim*2, 'individuals must be more than 2*dim'

        # 1. 対称性のチェック
        if not np.allclose(self.C, self.C.T, atol=1e-8):
            print(self.C)
            raise ValueError("Matrix is not symmetric. Ensure C is symmetric.")

        # 2. サイズが大きすぎる場合のチェック
        # 一般的に非常に大きい行列 (>1000x1000) は計算の収束に問題を起こす可能性があります。
        if self.C.shape[0] > 1000:
            raise ValueError("Matrix is too large for stable eigenvalue computation.")

        # 3. 実数のみで構成されているかチェック
        if not np.isrealobj(self.C):
            raise ValueError("Matrix contains complex elements. Ensure C is a real matrix.")

        # Cの固有値と固有ベクトルを取得
        try:
            self.eigvals, self.eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Eigenvalue computation did not converge.") from e

        # 固有値を範囲内に収める
        self.eigvals = np.clip(self.eigvals, 0.2, 0.95) # ! 

        # n次元空間の各軸に沿ったスケールを計算
        scales = np.sqrt(self.eigvals)

        yoyaku = 0
        for i in range(self.dim):
            self.individuals[yoyaku].location = self.centroid + self.sigma * self.eigvecs[i] * scales[i]
            np.clip(self.individuals[yoyaku].location, self.lower_search_bound[0], self.upper_search_bound[0], out=self.individuals[yoyaku].location)
            yoyaku += 1
            self.individuals[yoyaku].location = self.centroid - self.sigma * self.eigvecs[i] * scales[i]
            np.clip(self.individuals[yoyaku].location, self.lower_search_bound[0], self.upper_search_bound[0], out=self.individuals[yoyaku].location)
            yoyaku += 1
        
        self.scales = scales  # TODO: test
        
        Z = np.random.normal(0, 1, size=(self.lam-yoyaku, self.dim))
        BD = np.matmul(self.eigvecs, np.diag(scales))
        Y = np.matmul(BD, Z.T).T
        for i in range(yoyaku, self.lam):
            self.individuals[i].location = self.centroid + self.sigma * Y[i-yoyaku]
            np.clip(self.individuals[i].location, self.lower_search_bound[0], self.upper_search_bound[0], out=self.individuals[i].location)

        self.val_diff_count = 0

    def update_with_tpso_classed(self, gen, tpso_best_individual):
        """ 正規分布パラメータの更新
            X (np.ndarray): 個体群, shape==(self.lam, self.dim)
            fitnesses (np.ndarray): 適合度
            gen (int): 現在の世代数
        """
        
        """1. Selection and recombination"""
        old_centroid = self.centroid
        # self.old_centroid = self.centroid
        old_sigma = self.sigma

        X = np.array([individual.location for individual in self.individuals])
        fitnesses = np.array([individual.fitness for individual in self.individuals])
        # tPSOの最良個体の情報を追加
        if tpso_best_individual is not None:
            if np.max(fitnesses) < tpso_best_individual.fitness:
                fitnesses = np.append(fitnesses, tpso_best_individual.fitness)
                if tpso_best_individual.location.ndim == 1:
                    X = np.vstack((X, tpso_best_individual.location.reshape(1, -1)))
                else:
                    X = np.vstack((X, tpso_best_individual.location))
        #: fitnessが上位μまでのインデックスを抽出(降順で)
        elite_indices = np.argsort(fitnesses)[-self.mu:][::-1]

        X_elite = X[elite_indices, :]
        Y_elite = (X_elite - old_centroid) / old_sigma
        
        X_w = np.matmul(self.weights, X_elite)[0]
        Y_w = np.matmul(self.weights, Y_elite)[0]

        #: 正規分布中心の更新
        self.centroid = (1 - self.c_m) * old_centroid + self.c_m * X_w
        
        # Apply constraints
        np.clip(self.centroid, self.lower_search_bound[0], self.upper_search_bound[0], out=self.centroid)
        
        """ 2. Step-size control """
        diagD, B = np.linalg.eigh(self.C)
        diagD = np.sqrt(diagD)
        inv_diagD = 1.0 / diagD

        #: Note. 定義からnp.matmul(B, Z.T).T == np.matmul(C_, Y.T).T
        C_ = np.matmul(np.matmul(B, np.diag(inv_diagD)), B.T)

        new_p_sigma = (1 - self.c_sigma) * self.p_sigma
        new_p_sigma += np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.matmul(C_, Y_w)
        self.p_sigma = new_p_sigma

        E_normal = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21 * self.dim **2)) #:定数パラメータ
        self.sigma = self.sigma * np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.sqrt((self.p_sigma ** 2).sum()) / E_normal - 1)
        )
        
        # *Apply constraints
        self.sigma = np.array([self.sigma])
        np.clip(self.sigma, 3, 10, out=self.sigma)
        self.sigma = self.sigma[0]
        
        """ 3. Covariance matrix adaptation (CMA) """
        #: Note. h_σ(heaviside関数)はステップサイズσが大きいときにはCの更新を中断させる
        left = np.sqrt((self.p_sigma ** 2).sum()) / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (gen+1)))
        right = (1.4 + 2 / (self.dim + 1)) * E_normal
        hsigma = 1 if left < right else 0
        d_hsigma = (1 - hsigma) * self.c_c * (2 - self.c_c)

        #: p_cの更新
        new_p_c = (1 - self.c_c) * self.p_c
        new_p_c += hsigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * Y_w
        self.p_c = new_p_c

        #: Cの更新
        new_C = (1 + self.c_1 * d_hsigma - self.c_1 - self.c_mu) * self.C
        new_C += self.c_1 * np.outer(self.p_c, self.p_c)

        #: 愚直な実装（スマートな実装はdeapのcma.pyを参照)
        wyy = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            y_i = Y_elite[i]
            wyy += self.weights[0, i] * np.outer(y_i, y_i)
        new_C += self.c_mu * wyy

        # 正規化処理
        frobenius_norm = np.linalg.norm(new_C, 'fro')
        new_C = new_C / frobenius_norm
        self.C = new_C
    
    def update_current_best_fitness(self, individual=None):
        """現時点での最良の評価値(self.current_best_fitness)を計算する(TCMA-ES用)

        Args:
            fitness (float, optional): 今のベストと比較する用．絶対に現環境のものであること. Defaults to None.
        """
        if individual is not None:
            if self.parent_swarm.current_best_eval_env_num != self.problem.env_num or self.parent_swarm.current_best_fitness < individual.fitness:
                self.parent_swarm.current_best_fitness = individual.fitness
                self.parent_swarm.current_best_eval_env_num = self.problem.env_num
        else:
            # 全探索する
            tpso_best_fitness = max([individual.fitness for individual in self.parent_swarm.individuals if individual.eval_env_num == self.problem.env_num], default=None)
            cmaes_best_fitness = max([individual.fitness for individual in self.cmaes.individuals if individual.eval_env_num == self.problem.env_num], default=None)
            if cmaes_best_fitness is None or tpso_best_fitness > cmaes_best_fitness:
                self.parent_swarm.current_best_fitness = tpso_best_fitness
            else:
                self.parent_swarm.current_best_fitness = cmaes_best_fitness
            self.parent_swarm.current_best_eval_env_num = self.problem.env_num
    
    def update_cbest(self, individual=None):
        """cbestを更新する(tcmaes視点)
        individualが指定された場合，その個体がcbestを更新するか判定する
        individualが指定されない場合，現在の個体群からcbestを更新する
        
        Args:
            individual (Individual, optional): 更新する個体. Defaults to None.
        """
        
        old_cbest = self.parent_swarm.cbest
        
        if individual is not None:
            if individual.fitness > self.parent_swarm.cbest.fitness:
                self.cbest = individual
                if individual.kind == 'TCMAES':
                    self.cmaes.cbest = individual
                elif individual.kind == 'TPSO':
                    self.tpso_cbest = individual
        else:
            self.parent_swarm.tpso_cbest = max(
                (individual for individual in self.parent_swarm.individuals if individual.eval_env_num == self.problem.env_num),
                key=lambda x: x.fitness,
                default=None
            )
            
            self.cbest = max(
                (individual for individual in self.individual if individual.eval_env_num == self.problem.env_num),
                key=lambda x: x.fitness,
                default=None
            )
        
        if self.parent_swarm.tpso_cbest.fitness > self.cbest.fitness:
            self.parent_swarm.cbest = self.parent_swarm.tpso_cbest
        else:
            self.parent_swarm.cbest = self.cbest
        
        # cbestが変わっていたらTrueにする
        if self.parent_swarm.cbest != old_cbest:
            self.parent_swarm.is_updated_cbest = True
    
    def update_gbest(self, individual=None):
        # gbestの更新の必要は無いから代わりにcbestの更新
        self.update_cbest(individual)
        
        # current_best_fitnessの更新
        self.update_current_best_fitness(individual)

class Individual(BaseIndividual):
    def __init__(self, problem, location=None, evaluate=True):
        super().__init__(problem, location, evaluate)
        self.is_evaluated = False
        self.kind = 'TCMAES'