from abc import ABC, abstractmethod
import numpy as np

class BaseIndividual(ABC):
    def __init__(self, problem, location=None, evaluate=True):
        self.problem = problem
        if location is not None:
            self.location = location
        else:
            # ランダムに設定
            self.location = (problem.search_range[:, 1] - problem.search_range[:, 0]) * np.random.rand(problem.dim) + problem.search_range[:, 0] * np.ones(problem.dim)
        
        self.velocity = np.zeros(problem.dim)
        self.fitness = None
        self.eval_env_num = None  # change_detection == unknownの場合，これは評価の時しか参照できない
        self.pbest_location = self.location.copy()
        self.pbest_fitness = self.fitness
        self.pbest_eval_env_num = self.eval_env_num
        self.last_location = self.location.copy()
        self.last_fitness = None  # 評価直前のfitnessを記録
        self.evaled_order = ""  # そのイテレーションで評価された順番(複数回評価された場合は最後の値)
    
    def update_pbest(self):
        """自身のpbestを更新する．環境は関係ない
        """
        if self.pbest_fitness is None or self.fitness > self.pbest_fitness:
            self.pbest_location = self.location.copy()
            self.pbest_fitness = self.fitness
            self.pbest_eval_env_num = self.eval_env_num
    
    def update_gbest(self, swarm):
        # TODO: 使って無くない？base_swarmの方使ってる気がする
        # 条件に合う個体から fitness が最大のものを選ぶ(cbest)
        if swarm.cbest is None or swarm.cbest.eval_env_num != self.problem.env_num or  self.fitness > swarm.cbest.fitness:
            swarm.cbest = self
        
        # gbestの更新
        if swarm.gbest_fitness is None or self.fitness > swarm.gbest_fitness:
            swarm.gbest_location = self.location.copy()
            swarm.gbest_fitness = self.fitness
            swarm.gbest_eval_env_num = self.eval_env_num
        
        # swarm.current_best_fitnessを計算
        # * swarm.current_best_fitnessが現環境のものでない場合，必ず更新
        swarm.update_current_best_fitness(self.fitness)