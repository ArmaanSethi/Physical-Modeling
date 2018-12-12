import numpy as np
import random
import matplotlib.pyplot as plt
#=========================================
# demonstrator for genetic algorithms: 
# How many guesses do we need to find 
# a specific sentence?
#=========================================
# mutates a few genes, depending on mutation rate
def mutate(gen_parent,rate_mutation):
    gen_child = ''
    for char in gen_parent:
        if (random.random() < rate_mutation):
            char = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        gen_child = gen_child+char
    return gen_child

#=========================================
# measures the fitness
def fitness(gen_curr,gen_goal):
    return sum(gen_curr[i] == gen_goal[i] for i in range(len(gen_goal)))

#=========================================
# evolve(gen_str,gen_goal,nchild,fFIT)
# evolves generations until goal is met
#-----------------------------------------
def evolve(gen_goal,nchild,fFIT,rate_mutation):
    ngen     = len(gen_goal)
    gen_curr = ''
    for i in range(ngen):
        gen_curr = gen_curr+random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
    gen_best = ''
    gen_next = ''
    fit_goal = fFIT(gen_goal,gen_goal) 
    fit_curr = fFIT(gen_curr,gen_goal)
    fit_next = -1
    fit_best = -1

    igen     = 0
    while (fit_goal != fit_curr):
        for i in range(nchild):
            gen_next = mutate(gen_curr,rate_mutation)  
            fit_next = fFIT(gen_next,gen_goal)
            if (fit_next >= fit_curr):
                fit_best = fit_next
                gen_best = gen_next
        gen_curr = gen_best
        fit_curr = fit_best
        igen     = igen + 1 
        print("igen=%5i fit=%13.5e gen=%s" % (igen,float(fit_curr)/float(fit_goal),gen_curr))
    return igen

#======================================
def main():
    nchild        = 30
    gen_goal      = "BEDECKE DEINEN HIMMEL ZEUS MIT WOLKENDUNST UND UEBE KNABEN GLEICH DER DISTELN KOEPFT AN EICHEN DICH UND BERGESHOEHN"
    fFIT          = fitness
    rate_mutation = 0.01

    it            = evolve(gen_goal,nchild,fFIT,rate_mutation)
   
    print("Done in %7i iterations for 26^%i possible combinations" % (it,len(gen_goal)))

    
#======================================
main()
    
