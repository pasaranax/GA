'''
@author: Mikhail Bulygin
@email: pasaranax@gmail.com
'''
from random import randint, choice, uniform, sample
from time import time
try:
    import numpy as np
    from matplotlib import pyplot as plt
    matplot_installed = True
except ImportError:
    matplot_installed = False

class GA():
    def __init__(self, evaluator, bounds=None, num_genes=None, init=None, steps=50, stop_spread=None, stop_fitness=None, stagnation=0,
                 population_limit=20, survive_coef=0.25, productivity=4, default_step=0.01, default_bounds=(-100, 100),
                 mutagen="1_step", cata_mutagen="full_step", verbose=True, plot=False):
        '''
        :param: evaluator - фитнес-функция
        :param: bounds - границы значений генов и шаг для изменения, tuple(left, right, step), одно значение или список для каждого гена
        :param: num_genes - количество параметров (генов)
        :param: init - набор генов для особи, которая добавляется в начальную популяцию (альфа-самец)
        :param: steps - количество итераций (поколений)
        :param: stop_spread - если разброс результатов лучших особей менее чем stop_spread, завершить цикл
        :param: stop_fitness - если fitness достигнет stop_fitness, завершить цикл
        :param: stagnation - если количество итераций с одинаковым спредом достигнет stagnation, вызвать катаклизм (усиленная мутация)
        :param: population_limit - размер популяции
        :param: survive_coef - процент выживших (лучших) после каждой итерации
        :param: productivity - количество потомков на каждую выжившую особь
        :param: default_bounds - границы по умолчанию
        :param: default_step - шаг по умолчанию для мутации *_step
        :param: mutagen - тип мутации (1_step, full_step, 1_random, full_random, 1_change, full_change)
                          1_step - менять один ген на размер шага, full_step - так же менять все гены,
                          1_random - менять один ген на случайное число в диапазоне bounds, full_random - так же менять все гены,
                          1_change - менять один ген на 0-10% (случайно), full_change - так же менять все гены
        :param: cata_mutagen - тип мутации при катаклизме
        :param: verbose - уровень вывода в консоль
        :param: plot - рисовать график результатов (необходим matplotlib, может замедлить работу)
        '''
        assert type(bounds) is list or (type(bounds) is tuple and type(num_genes) is int)
        self.evaluator = evaluator
        self.init = init
        self.steps = steps
        self.stop_spread = stop_spread
        self.stop_fitness = stop_fitness
        self.stagnation = stagnation
        self.population_limit = population_limit
        self.survive_coef = survive_coef
        self.productivity = productivity
        self.default_step = default_step
        self.mutagen = mutagen
        self.cata_mutagen = cata_mutagen
        self.verbose = verbose
        self.plot = plot
        
        self.best_ever = None  # место для самого лучшего
        self.fitness = []  # сохраняем рейтинги для анализа
        self.spreads = []  # сохраняем спреды для анализа
        
        if type(bounds) is list:
            self.bounds = bounds
        elif type(bounds) is tuple and num_genes:
            self.bounds = self.gen_bounds(bounds[0], bounds[1], bounds[2], num_genes)
        elif not bounds:
            self.bounds = self.gen_bounds(default_bounds[0], default_bounds[1], 
                                          default_step, num_genes)
        
        if matplot_installed and plot:
            self.prepare_plot()
        else:
            plot = False
        
    def evolve(self, steps=None):
        '''
        Запустить эволюцию
        :param: step - количество шагов можно указать здесь
        '''
        if steps:
            self.steps = steps
        t = time()
        newborns = []  # новорожденные без фитнеса
        if self.init:
            newborns.append(self.init)
        for i in range(self.steps):
            ti = time()
            population = self.generate_population(newborns)
            if not self.best_ever:
                self.best_ever = population[0]
            best = self.survive(population)            
            newborns = self.crossover(best)
            
            self.best_ever = max(self.best_ever, best[0], key=lambda i: i[1])
            self.spreads.append(self.best_ever[1] - min(best, key=lambda i: i[1])[1])
            self.fitness.append([i[1] for i in population])

            if self.verbose:
                elapsed = time()-t
                remaining = (time()-ti)*(self.steps-i)
                print("- Step {:d} / {:d} results: best: {:.3f}, spread: {:.3f}, elapsed: {:.0f}m {:.0f}s, remaining: {:.0f}m {:.0f}s".
                      format(i+1, self.steps, self.best_ever[1], self.spreads[-1],
                             elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))
            
            # условие катаклизма
            if self.stagnation > 1 and len(set(self.spreads[-self.stagnation:])) == 1:
                newborns = self.cataclism(population, self.cata_mutagen)

            # условия досрочного завершения
            if self.stop_spread != None and self.spreads[-1] <= self.stop_spread:
                if self.verbose >= 1:
                    print("- Evolution complete: spread = {:.3f} <= {:.3f}".format(self.spreads[-1], self.stop_spread))
                break
            if self.stop_fitness != None and self.best_ever[1] >= self.stop_fitness:
                if self.verbose >= 1:
                    print("- Evolution complete: best fitness = {:.3f} <= {:.3f}".format(self.best_ever[1], self.stop_fitness))
                break
            
            if self.plot:
                self.draw()
        if self.plot:
            plt.ioff()
            plt.show()
        return self.best_ever
    
    def generate_population(self, newborns):
        population = []
        # добавляем мутации новорожденным
        for indiv in newborns:
            indiv = self.mutate(indiv, self.mutagen)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))
            
        # создаем случайных особей, если есть места в популяции
        for _ in range(self.population_limit - len(newborns)):
            indiv = []
            for bounds in self.bounds:
                if self.mutagen.endswith("random") or self.mutagen.endswith("change"):
                    gene = uniform(bounds[0], bounds[1])
                elif self.mutagen.endswith("step"):
                    step = bounds[2]
                    gene = choice(frange(bounds[0], bounds[1]+step, step))
                indiv.append(gene)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))
        return population
    
    def survive(self, population):
        num_survivors = int(self.population_limit * self.survive_coef)
        best = sorted(population, key=lambda i: -i[1])[:num_survivors]
        return best
    
    def crossover(self, best):
        newborns = []
        for _ in range(len(best) * self.productivity):
            dad, mom = sample(best, 2)
            child = []
            for gene_m, gene_f in zip(dad[0], mom[0]):  # извлекаем геном
                gene = choice((gene_m, gene_f))
                child.append(gene)
            newborns.append(child)
        return newborns

    def mutate(self, indiv, mutagen):
        if mutagen == "1_random":
            gene_id = randint(0, len(indiv)-1)
            indiv[gene_id] = uniform(self.bounds[gene_id][0], self.bounds[gene_id][1])
        elif mutagen == "full_random":
            for gene_id in range(len(indiv)):
                indiv[gene_id] = uniform(self.bounds[gene_id][0], self.bounds[gene_id][1])
        elif mutagen == "1_change":
            gene_id = randint(0, len(indiv)-1)
            while True:
                coef = uniform(0.9, 1.1)
                if self.bounds[gene_id][0] <= indiv[gene_id] * coef <= self.bounds[gene_id][1]:
                    indiv[gene_id] *= coef
                    break
        elif mutagen == "full_change":
            for gene_id in range(len(indiv)):
                while True:
                    coef = uniform(0.9, 1.1)
                    if self.bounds[gene_id][0] <= indiv[gene_id] * coef <= self.bounds[gene_id][1]:
                        indiv[gene_id] *= coef
                        break
        elif mutagen == "1_step":
            gene_id = randint(0, len(indiv)-1)
            while True:
                step = self.bounds[gene_id][2]
                step = choice([-step, step])
                if self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]:
                    indiv[gene_id] += step
                    break
        elif mutagen == "full_step":
            for gene_id in range(len(indiv)):
                while True:
                    step = self.bounds[gene_id][2]
                    step = choice([-step, step])
                    if self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]:
                        indiv[gene_id] += step
                        break
        return indiv
    
    def cataclism(self, population, mutagen):
        post_population = []
        for indiv, _fitness in population:
            post_population.append(self.mutate(indiv, mutagen))
        if self.verbose >= 1:
            print("- Cataclism occured because stagnation {}: {}".format(self.stagnation, mutagen))
        return post_population
    
    def gen_bounds(self, left, right, step, num):
        return [(left, right, step) for _ in range(num)]
    
    def prepare_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2, sharex=True)
        self.ax[0].set_ylabel("Fitness")
        self.ax[1].set_ylabel("Spread")
        self.ax[0].set_xlim([0, self.steps])
        plt.pause(0.0001)
        self.avg = []
    
    def draw(self):
        self.avg.append(sum(self.fitness[-1]) / len(self.fitness[-1]))
        x = np.asarray([range(len(self.fitness)) for _ in range(len(self.fitness[0]))])
        y = np.transpose(self.fitness)
        self.ax[0].scatter(x=x, y=y, color="blue", marker=".")
        self.ax[0].plot(self.avg, color="red", lw=1)
        self.ax[1].plot(self.spreads, color="green", lw=1)
#         low, high = 1, 1
#         for gen in self.fitness:
#             low = min(low, min(gen))
#             high = max(high, max(gen))
#         self.ax[0].set_ylim([low, high])
        plt.pause(0.001)

def frange(start, stop, step):
    flist = []
    while start < stop:
        flist.append(start)
        start += step
    return flist

def test_equation(x):
    '''Equation: a + 2b + 3c + 4d = 30'''
    z = x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3]
    ans = 30
    print("{:.1f} {:+.1f}*2 {:+.1f}*3 {:+.1f}*4 = {:.1f}".format(x[0], x[1], x[2], x[3], z), "- Solved!" if z == ans else "")
    return -abs(ans-z)
    
if __name__ == '__main__':
    ga = GA(test_equation, bounds=(-100, 100, 1), num_genes=4, stop_fitness=0, stagnation=3, plot=True)
    result = ga.evolve()
    print("Best solution:", result)
    
    