# -*- coding: utf-8 -*-
import numpy as np
import csv

rule_length = 15

def encode(x):
    temperatur = ['Rendah', 'Normal', 'Tinggi'] 
    waktu = ['Pagi', 'Siang', 'Sore', 'Malam'] 
    langit = ['Cerah', 'Berawan', 'Rintik', 'Hujan']  
    kelembaban = ['Rendah', 'Normal', 'Tinggi']  
    terbang = ['Tidak', 'Ya']
    
    
    arr_kromosom = []

    # temperatur
    temp = np.zeros(len(temperatur))
    temp[temperatur.index(x[0])] = 1
    arr_kromosom.append(temp)

    # langit
    lgt = np.zeros(len(langit))
    lgt[langit.index(x[2])] = 1
    arr_kromosom.append(lgt)

    # waktu
    wkt = np.zeros(len(waktu))
    wkt[waktu.index(x[1])] = 1
    arr_kromosom.append(wkt)
    
    # kelembaban
    lembab = np.zeros(len(kelembaban))
    lembab[kelembaban.index(x[3])] = 1
    arr_kromosom.append(lembab)
    
     # terbang
    if(x[4] != ''):  
        arr_kromosom.append(terbang.index(x[4]))
    
    return arr_kromosom

def LoadData(data_latih='data_latih_opsi_1.csv', data_uji='data_uji_opsi_1.csv'):
    arr_latih = []
    arr_uji = []
    with open(data_latih, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        arr_latih = list(reader)
    
    with open(data_uji, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        arr_uji = list(reader)
        
    return [encode(x) for x in arr_latih], [encode(x) for x in arr_uji]

latih, uji = LoadData()

class Kromosom:
    def __init__(self, kromosom=None):
        if kromosom == None:
            bil = np.random.randint(2, 5)
            self.kromosom = np.random.randint(2, size=rule_length*bil)
        else:
            self.kromosom = np.array(kromosom)
    
    def answer(self):
        num_rule = self.kromosom.shape[0]
        rules = []
        for i in range(num_rule):
            kr = self.kromosom[i*15:(i+1)*15]
            h = [kr[:3], kr[3:7], kr[7:11], kr[11:14], kr[-1]]
            rules.append(h)
        return rules
    
    def Fitness(self):
        ru = self.answer()
        bil = 0
        predict_value = self.Predict(latih)
        for data, label in zip(latih, predict_value):
            if value == data[-1]:
                bil += 1
        return bil/len(latih)
    
    def Mutasi(self, mts=None):
        if mts == None:
            mts = 1/self.kromosom.shape[0]
        
        if np.random.rand() < mts:
            mk = np.random.randint(2, size=self.kromosom.shape[0])
            for i in range(self.kromosom.shape[0]):
                if mk[i] == 1:
                    self.kromosom[i] = 1 if self.kromosom[i] == 0 else 0
    
    def Predict(self, uji):
        ru = self.answer()
        values = []
        for data in uji:
            found = False
            for rule in ru:
                truth = True
                for r1, r2 in zip(data[:-1], rule):
                    p = int("".join([ str(y) for y in r1.astype('uint8') ]), 2)
                    q = int("".join([ str(y) for y in r2.astype('uint8') ]), 2)
                    r = p & q
                    truth = truth and (r == p)
                if truth:
                    value = rule[-1]
                    found = True
                    values.append(value)
                    break
            if not found:
                values.append(0)
        return values

class Population:
    def __init__(self, num=100):
        self.population = []
        for i in range(num):
            self.population.append(Kromosom())
     
    def RoulleteWheel(self):
        total_prob = np.sum([kromosom.Fitness() for kromosom in self.population])
        offset = 0
        selected = self.population[0]
        r = np.random.rand()
        for i, kromosom in enumerate(self.population):
            offset += kromosom.Fitness()
            if offset > r:
                break
            selected = kromosom
        return selected
    
    def Crossover(self, krom1, krom2, pc=1):
        if np.random.rand() > pc:
            return krom1, krom2
        pil = []
        while len(pil) == 0:
            p1 = np.random.randint(1, krom1.kromosom.shape[0]-1, size=2)
            if(p1[1] < p1[0]):
                p1[0], p1[1] = p1[1], p1[0]
            t1 = p1[0] % 15
            t2 = p1[1] % 15

            jml_aturan = krom2.kromosom.shape[0] 

            for i in range(jml_aturan):
                for j in range(i, jml_aturan):
                    x = i*15+t1
                    y = j*15+t2
                    if x > y:
                        continue
                    pil.append([x, y])

        p2 = pil[np.random.randint(0, len(pil))]

        offspr1 = [*krom1.kromosom[:p1[0]], *krom2.kromosom[p2[0]:p2[1]], *krom1.kromosom[p1[1]:]]
        offspr2 = [*krom2.kromosom[:p2[0]], *krom1.kromosom[p1[0]:p1[1]], *krom2.kromosom[p2[1]:]]
        kromosom1 = Kromosom(offspr1)
        kromosom2 = Kromosom(offspr2)
        return kromosom1, kromosom2

    def FindBest(self):
        bil = 0
        for i in range(1, len(self.population)):
            if self.population[i].Fitness() > self.population[bil].Fitness():
                bil = i
        return self.population[bil]

population = Population(100)

for generation in range(20):
    arr_population = []
    print('Generasi ', (generation+1))
    while len(arr_population) < len(population.population):
      #parent
        parent1 = population.RoulleteWheel()
        parent2 = population.RoulleteWheel()
      #child
        child1, child2 = population.Crossover(parent1, parent2, 1)
        child1.Mutasi()
        child2.Mutasi()

        arr_population.append(child1)
        arr_population.append(child2)
    population.population = arr_population
    best = population.FindBest()
    print('Best Fitness: ', best.Fitness())

akurasi = best.Fitness()
print('Akurasi: ', akurasi)
value = best.Predict(uji)
print('value: ', value)

print('Kromosom')
print(best.kromosom)

np.savetxt('hasil.txt', value, fmt='%i')
