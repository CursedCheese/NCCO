from abc import ABC, abstractmethod
import numpy as np

class BaseSwarm(ABC):
    """
    cbest: 現環境で最良の個体
    gbest_location: 歴代で最良の座標
    gbest_fitness: 歴代で最良の適応度
    gbest_eval_env_num: 歴代で最良の座標/適応度を記録した環境番号
    """
    def __init__(self, problem):
        self.individuals = []
        self.cbest = None
        self.problem = problem
        self.gbest_location = None  # 歴代(現環境)最良の座標
        self.gbest_fitness = None
        self.gbest_eval_env_num = None  # 歴代最良を記録した環境番号
        self.current_best_fitness = None  # 現時点(現環境)での最良の評価値, 評価指標の計算に使用
        self.current_best_eval_env_num = None  # self.current_best_fitnessを記録した環境番号
    
    @abstractmethod
    def update_gbest(self, individual=None):
        """Swarmのgbestを更新する
        """
        # individualを指定した場合，単純に比較
        if individual is not None:
            if self.gbest_fitness is None or individual.pbest_fitness > self.gbest_fitness:
                self.gbest_location = individual.pbest_location.copy()
                self.gbest_fitness = individual.pbest_fitness
                self.gbest_eval_env_num = individual.eval_env_num
            return
        
        # individualを指定しない場合，全個体からgbestを更新
        valid_individuals = [individual for individual in self.individuals if individual.pbest_fitness is not None]
        if len(valid_individuals) == 0:
            gbest_index = 0
        else:
            gbest_index = np.argmax([individual.pbest_fitness for individual in valid_individuals])
        self.gbest_location = self.individuals[gbest_index].pbest_location.copy()
        self.gbest_fitness = self.individuals[gbest_index].pbest_fitness
        self.gbest_eval_env_num = self.individuals[gbest_index].pbest_eval_env_num
    
    def update_current_best_fitness(self, individual=None, fitness=None):
        """現時点での最良の評価値(self.current_best_fitness)を計算する
        この値は評価指標の計算に使用される
        fitnessを指定する場合は，そのfitnessは必ず現環境のものであること
        """
        # individualが指定されている場合，その個体と比較する
        if individual is not None:
            # 今の最良評価値が評価されたのは現環境かどうか
            if self.current_best_eval_env_num == self.problem.env_num:
                if self.current_best_fitness < individual.fitness:
                    self.current_best_fitness = individual.fitness
            # 現在の最良評価値が現環境のものではないなら問答無用で更新
            else:
                self.current_best_fitness = individual.fitness
                self.current_best_eval_env_num = self.problem.env_num
            return
        
        if fitness is not None:
            if self.current_best_eval_env_num != self.problem.env_num or self.current_best_fitness < fitness:
                self.current_best_fitness = fitness
                self.current_best_eval_env_num = self.problem.env_num
            return
        
        # fitnessが指定されていない場合，全個体から求める
        if self.gbest_eval_env_num == self.problem.env_num:
            if self.gbest_fitness > self.cbest.fitness:
                self.current_best_fitness = self.gbest_fitness
            else:
                self.current_best_fitness = self.cbest.fitness
        else:
            self.current_best_fitness = self.cbest.fitness
        self.current_best_eval_env_num = self.problem.env_num
    
    def update_cbest(self, individual=None):
        """cbestを更新する
        cbestとは現在最良の解. gbestとは異なる. gbestは歴代最も良い評価を記録した座標と評価値. gbestなら現在そこに個体は居ない可能性もある

        Args:
            individual (Individual, optional): 比較対象. Defaults to None.
        """
        # TODO: 誰も使ってない(DMSFPSO, SPSOAPADで使ってみる)
        # 引数の個体がある場合，現在のcbestと比較
        if individual is not None:
            if self.cbest is None or individual.fitness > self.cbest.fitness:
                self.cbest = individual
            return
        
        # 引数が無かった場合，全個体と比較
        self.cbest = max(
            (individual for individual in self.individuals if individual.eval_env_num == self.problem.env_num),
            key=lambda x: x.fitness,
            default=self.individuals[0]
        )
    