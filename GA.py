'''
@author: Mikhail Bulygin
@email: pasaranax@gmail.com
'''
from random import randint, choice, uniform, sample, shuffle
from time import time
from math import exp, isclose
from json import dump, load
from threading import Thread
try:
    import numpy as np
    import matplotlib.pyplot as plt
    matplot_installed = True
except ImportError:
    matplot_installed = False

class GA():
    def __init__(self, evaluator, bounds=None, num_genes=None, init=None, steps=100, stop_fitness=None, time_limit=None, 
                 stagnation=None, population_limit=20, survive_coef=0.25, productivity=4, cross_type="split", 
                 mutate_genes=1, cata_mutate_genes=2, mutagen="1_step", cata_mutagen="1_step", 
                 autosave="population.json", verbose=True, plot=False):
        '''
        :param evaluator: фитнес-функция
        :param bounds: границы значений генов и шаг для изменения, tuple(left, right, step), одно значение или список для каждого гена
        :param num_genes: количество параметров (генов)
        :param init: набор генов для особи, которая добавляется в начальную популяцию (альфа-самец)
        :param steps: количество итераций (поколений)
        :param stop_fitness: если fitness достигнет stop_fitness, завершить цикл
        :param time_limit: ограничение по времени в секундах, после которого стоит завершить эволюцию
        :param stagnation: если количество итераций с одинаковым спредом достигнет stagnation, вызвать катаклизм (усиленная мутация)
        :param population_limit: размер популяции
        :param survive_coef: процент выживших (лучших) после каждой итерации
        :param productivity: количество потомков на каждую выжившую особь
        :param cross_type: тип скрещивания (split, random)
                            split - первая половина генов от папы, вторая половина от мамы
                            random - в случайном порядке от родителей
                            uniq_split - аналогичен split, но повторяющиеся значения заменяются случайными (все гены уникальны)
        :param mutate_genes: сколько генов модифицировать в мутациях 1_*
        :param cata_mutate_genes: сколько генов модифицировать в мутациях 1_* при катаклизме
        :param mutagen: тип мутации (1_step, full_step, 1_random, full_random, 1_change, full_change)
                          1_step - менять один ген на размер шага, full_step - так же менять все гены,
                          1_random - менять один ген на случайное число в диапазоне bounds, full_random - так же менять все гены,
                          1_change - менять один ген на 0-10% (случайно), full_change - так же менять все гены
                          swap - поменять местами 2 гена (используйте только если все гены имеют одинаковые границы bounds)
                          swap_near - поменять местами 2 соседних гена
        :param cata_mutagen: тип мутации при катаклизме
        :param verbose: уровень вывода в консоль
        :param plot: рисовать график результатов (необходим matplotlib, может замедлить работу)
        '''
        assert type(bounds) is list or (type(bounds) is tuple and type(num_genes) is int)
        self.evaluator = evaluator
        self.init = init
        self.steps = steps
        self.stop_fitness = stop_fitness
        self.time_limit = time_limit
        self.stagnation = stagnation
        self.population_limit = population_limit
        self.survive_coef = survive_coef
        self.productivity = productivity
        self.cross_type = cross_type
        self.mutate_genes = mutate_genes
        self.cata_mutate_genes = cata_mutate_genes
        self.mutagen = mutagen
        self.cata_mutagen = cata_mutagen
        self.autosave = autosave
        self.verbose = verbose
        self.plot = plot
        
        self.best = []  # История лучших
        self.fitness = []  # сохраняем рейтинги для анализа
        self.spreads = []  # сохраняем спреды для анализа
        default_step = 0.01
        default_bounds = (-100, 100)
        
        if type(bounds) is list:
            self.bounds = bounds
        elif type(bounds) is tuple and num_genes:
            try:
                self.bounds = [(bounds[0], bounds[1], bounds[2])] * num_genes
            except IndexError:
                self.bounds = [(bounds[0], bounds[1], default_step)] * num_genes
        elif not bounds:
            self.bounds = self.gen_bounds(default_bounds[0], default_bounds[1], 
                                          default_step, num_genes)
        
        if matplot_installed and plot:
            self.plotter = Plotter()
            self.plotter.start()
        else:
            self.plot = False
        
    def adopt(self, file, steps=None):
        '''
        Загрузить популяцию из файла и продолжить эволюцию
        :param file: file-like json-файл с готовой популяцией
        '''
        self.file = file
        newborns = load(file)
        return self.evolve(steps, newborns)
        
    def evolve(self, steps=None, newborns=None):
        '''
        Запустить эволюцию
        :param steps: количество шагов можно указать здесь
        :param newborns: продолжить эволюцию готовой популяции
        '''
        if steps:
            self.steps = steps
        t = time()
        if type(newborns) != list:
            newborns = []  # новорожденные без фитнеса
        if self.init:
            newborns.append(self.init)
        best_ever = None
        for i in range(self.steps):
            ti = time()
            population = self.generate_population(newborns)  # популяция с фитнесом
            survivors = self.survive(population)            
            newborns = self.crossover(survivors)
            
            self.best.append(survivors[0])
            self.fitness.append([i[1] for i in population])
            if not best_ever:
                best_ever = self.best[-1]
            else:
                best_ever = max(best_ever, self.best[-1], key=lambda i: i[1])

            if self.verbose:
                elapsed = time()-t
                remaining = (time()-ti)*(self.steps-i)
                print("- Step {:d} / {:d} results: best: {:.3f}, elapsed: {:.0f}m {:.0f}s, remaining: {:.0f}m {:.0f}s".
                      format(i+1, self.steps, best_ever[1],
                             elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))
            
            # сохраняем популяцию в файл
            if self.autosave:
                dump(newborns, open(self.autosave, "w"), separators=(",",":"))

            # условие катаклизма
            if self.stagnation:
                best_fitness = [i[1] for i in self.best_ever[-self.stagnation:]]
                if len(best_fitness) == self.stagnation and len(set(best_fitness)) == 1:
                    newborns = self.cataclysm(population)
            
            # условия досрочного завершения
            if self.stop_fitness != None and self.best_ever[-1][1] >= self.stop_fitness:
                if self.verbose >= 1:
                    print("- Evolution completed: best fitness = {:.3f} <= {:.3f}".format(self.best_ever[-1][1], self.stop_fitness))
                break
            
            if self.time_limit and time() - t >= self.time_limit:
                if self.verbose >= 1:
                    print("- Evolution completed: time is out: {} s.".format(self.time_limit))
                break
            
            if self.plot:
                self.plotter.update(self.fitness, self.spreads)
        if self.verbose >= 1:
            print("Best: {} - {}".format(best_ever[1], best_ever[0]))
        if self.plot:
            self.plotter.stop()
            self.plotter.join()
        return best_ever
    
    def generate_population(self, newborns):
        population = []
        # добавляем мутации новорожденным
        for indiv in newborns:
            indiv = self.mutate(indiv, self.mutagen, self.mutate_genes)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))
            
        # создаем случайных особей, если есть места в популяции
        for _ in range(self.population_limit - len(newborns)):
            indiv = []
            if "random" in self.mutagen or "change" in self.mutagen:
                for bounds in self.bounds:
                    gene = uniform(bounds[0], bounds[1])
                    indiv.append(gene)
            elif "step" in self.mutagen:
                for bounds in self.bounds:
                    step = bounds[2]
                    gene = choice(frange(bounds[0], bounds[1]+step, step))
                    indiv.append(gene)
            elif "swap" in self.mutagen:
                bounds = self.bounds[0]
                step = bounds[2]
                indiv = frange(bounds[0], bounds[1]+step, step)
                shuffle(indiv)
            fitness = self.evaluator(indiv)
            population.append((indiv, fitness))
            newborns.append(indiv)
        return population
    
    def survive(self, population):
        num_survivors = int(self.population_limit * self.survive_coef)
        best = sorted(population, key=lambda i: -i[1])[:num_survivors]
        return best
    
    def crossover(self, best):
        newborns = []
        for _ in range(len(best) * self.productivity):
            dad, mom = sample(best, 2)
            dad, mom = dad[0], mom[0]  # только геном без фитнеса
            child = []
            if self.cross_type == "random":
                for gene_m, gene_f in zip(dad, mom):  # извлекаем геном
                    gene = choice((gene_m, gene_f))
                    child.append(gene)
            elif self.cross_type == "split":
                split = len(dad) // 2
                child = dad[:split] + mom[split:]
            elif self.cross_type == "uniq_split":
                split = len(dad) // 2
                child = dad[:split] + mom[split:]
                bounds = self.bounds[0]
                step = bounds[2]
                for i, gene in enumerate(child):
                    # если ген с таким значением уже есть, генерируем случайное значение
                    if gene in child[:i]:
                        while True:  #TODO: сделать, чтобы цикл не был бесконечным
                            gene = choice(frange(bounds[0], bounds[1]+step, step))
                            if gene not in child:
                                child[i] = gene
                                break
            elif self.cross_type == "uniq_random":
                for gene_m, gene_f in zip(dad, mom):  # извлекаем геном
                    gene = choice((gene_m, gene_f))
                    child.append(gene)
                bounds = self.bounds[0]
                step = bounds[2]
                for i, gene in enumerate(child):
                    # если ген с таким значением уже есть, генерируем случайное значение
                    if gene in child[:i]:
                        while True:  #TODO: сделать, чтобы цикл не был бесконечным
                            gene = choice(frange(bounds[0], bounds[1]+step, step))
                            if gene not in child:
                                child[i] = gene
                                break
                
            newborns.append(child)
        return newborns

    def mutate(self, indiv, mutagen, mutate_genes=None):
        if mutagen == "1_random":
            gene_ids = [randint(0, len(indiv)-1) for _ in range(mutate_genes)]
            for gene_id in gene_ids:
                gene_id = randint(0, len(indiv)-1)
                indiv[gene_id] = uniform(self.bounds[gene_id][0], self.bounds[gene_id][1])
        elif mutagen == "full_random":
            for gene_id in range(len(indiv)):
                indiv[gene_id] = uniform(self.bounds[gene_id][0], self.bounds[gene_id][1])
        elif mutagen == "1_change":
            gene_ids = [randint(0, len(indiv)-1) for _ in range(mutate_genes)]
            for gene_id in gene_ids:
                while True:  #TODO: сделать, чтобы цикл не был бесконечным
                    coef = uniform(0.9, 1.1)
                    if self.bounds[gene_id][0] <= indiv[gene_id] * coef <= self.bounds[gene_id][1]:
                        indiv[gene_id] *= coef
                        break
        elif mutagen == "full_change":
            for gene_id in range(len(indiv)):
                while True:  #TODO: сделать, чтобы цикл не был бесконечным
                    coef = uniform(0.9, 1.1)
                    if self.bounds[gene_id][0] <= indiv[gene_id] * coef <= self.bounds[gene_id][1]:
                        indiv[gene_id] *= coef
                        break
        elif mutagen == "1_step":
            gene_ids = [randint(0, len(indiv)-1) for _ in range(mutate_genes)]
            for gene_id in gene_ids:
                gene_id = randint(0, len(indiv)-1)
                while True:  #TODO: сделать, чтобы цикл не был бесконечным
                    step = self.bounds[gene_id][2]
                    step = choice([-step, step])
                    if self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]:
                        indiv[gene_id] += step
                        break
        elif mutagen == "full_step":
            for gene_id in range(len(indiv)):
                while True:  #TODO: сделать, чтобы цикл не был бесконечным
                    step = self.bounds[gene_id][2]
                    step = choice([-step, step])
                    if self.bounds[gene_id][0] <= indiv[gene_id] + step <= self.bounds[gene_id][1]:
                        indiv[gene_id] += step
                        break
        elif mutagen == "swap":
            for _ in range(mutate_genes):
                gene_id_a, gene_id_b = sample(range(len(indiv)), 2)
                indiv[gene_id_a], indiv[gene_id_b] = indiv[gene_id_b], indiv[gene_id_a]
        elif mutagen == "swap_near":
            for _ in range(mutate_genes):
                gene_id = randint(0, len(indiv)-2)
                indiv[gene_id], indiv[gene_id+1] = indiv[gene_id+1], indiv[gene_id]            
        return indiv
    
    def cataclysm(self, population):
        post_population = []
        for indiv, _fitness in population:
            post_population.append(self.mutate(indiv, self.cata_mutagen, self.cata_mutate_genes))
        if self.verbose >= 1:
            print("- Cataclysm occured because of stagnation {} steps: {} ({} genes)".
                  format(self.stagnation, self.cata_mutagen, self.cata_mutate_genes))
        return post_population
    
    
class Plotter(Thread):
    def __init__(self):
        Thread.__init__(self)
#         self.daemon = True
        
    def update(self, fitness, spreads):
        self.fitness = fitness
        self.spreads = spreads
        
    def stop(self):
        self.running = False
        
    def run(self):
        self.running = True
        self.fitness = []
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        step = 0
        while True:
            if len(self.fitness) > step:
                step = len(self.fitness)
                avg = []
                for f in self.fitness:
                    avg.append(sum(f) / len(f))
                
                x = np.asarray([range(len(self.fitness)) for _ in range(len(self.fitness[0]))])
                y = np.transpose(self.fitness)
                
                ax.clear()
                ax.set_xlim([0, step])
                ax.scatter(x=x, y=y, color="blue", marker=".")
                ax.plot(avg, color="red", lw=1)
            plt.pause(0.000001)
            if not self.running and not plt.fignum_exists(1):
                plt.close(fig)
                break
    

def frange(start, stop, step):
    flist = []
    while start < stop:
        flist.append(start)
        start += step
    return flist

def example_diophante(x):
    '''Equation: a + 2b + 3c + 4d = 30'''
    a, b, c, d = x
    z = a + 2*b + 3*c + 4*d
    ans = 30
    print("a={:3.0f} b={:3.0f} c={:3.0f} d={:3.0f} z={:3.0f}".format(a, b, c, d, z), "- Solved!" if z == ans else "")
    return -abs(ans-z)
    
def example_rosenbrock(x):
    '''f(x1, x2) = (1 - x1)**2 + 100(x2 - x1**2)**2'''
    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    print("x1={:.3f}, x2={:.3f}, f={:.3f} {}".format(x[0], x[1], f, "- Close enough!" if isclose(f, 0, abs_tol=1e-6) else ""))
    return -abs(f)

def example_powell(x):
    '''f(x1, x2, x3, x4) = (x1 + 10*x2)**2 + 5(x3 - x4)**2 + (x2 - 2*x3)**4 + 10(x1 - x4)**4'''
    x1, x2, x3, x4 = x
    f = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4
    print("x1={:.3f}, x2={:.3f}, x3={:.3f}, x4={:.3f}, f={:.3f} {}".format(x1, x2, x3, x4, f, "- Close enough!" if isclose(f, 0, abs_tol=1e-6) else ""))
    return -abs(f)

def example_2dexp(x):
    '''f(x1, x2) = SUM(a, (e**(-ax1) - e**(-ax2)) - (e**(-a) - e**(-10a)))**2
    for a in frange(0.1, 1, 0.1)'''
    x1, x2 = x
    f = sum([((exp(-a*x1) - exp(-a*x2)) - (exp(-a) - exp(-10*a)))**2 for a in frange(0.1, 1, 0.1)])
    print("x1={:.3f}, x2={:.3f}, f={:.3f} {}".format(x1, x2, f, "- Close enough!" if isclose(f, 0, abs_tol=1e-6) else ""))
    return -abs(f)

if __name__ == '__main__':
    '''
    Пример:
    Попробуем найти одно из решений диофантова уравнения (с целочисленными корнями): a + 2b + 3c + 4d = 30.
    Фитнес-функция example_diophante получает на вход список предположительных корней уравнения и возвращает 
        отрицательное расстояние (чем больше фитнес, тем лучше) до его равенства (30).
    То есть при корнях являющихся решением, фитнес-функция вернет 0, во всех других случаях отрицательное число,
        которое чем ближе к нулю, тем больше наши корни похожи на решение.
    
    Параметры эволюции:
    steps = 40 - дадим эволюции не более 40 поколений
    stop_fitness = 0 - останавливаем эволюцию, когда функция вернула 0, значит решение найдено. Нужно указать с учетом точности.
    bounds = (-100, 100, 1) - предположим корни лежат где-то в диапазоне (-100, 100), шаг единица, поскольку корни целочисленные.
        От шага зависит точность поиска решения.
    num_genes = 4 - у нас 4 корня
    stagnation = 3 - если эволюция войдет в застой на 3 поколения, применяем катаклизм (более сильную мутацию)
    mutagen = "1_step" - у каждой особи (потенциального решения) при рождении создаем мутацию - 
        меняем один из параметров на размер шага
    cata_mutagen = "full_step" - если мы вошли в стагнацию, применяем катаклизм - меняем все параметры на размер шага
    population_limit = 10 - в каждом поколении будем тестировать 10 вариантов решения (особей)
    survive_coef = 0.2 - из каждого поколения выбираем 20% лучших решений (то есть 2 особи из 10 смогут оставить потомков)
    productivity = 4 - на каждую из двух выживших особей после скрещивания приходится 4 потомка, то есть в новом поколении 
        будет 8 потомков, остальные 2 места в популяции займут особи сгенерированные случайным образом
    plot = True - если установлен matplotlib, будем наблюдать эволюционный прогресс на графике
    '''
    ga = GA(example_diophante, bounds=(-100, 100, 1), num_genes=4, steps=40, stop_fitness=0, stagnation=3,
            population_limit=10, survive_coef=0.2, productivity=4, mutagen="1_step", cata_mutagen="full_step",
            plot=True)
    result = ga.evolve()
    print("Best solution:", result)
    
    