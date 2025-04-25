from neural import *

"""print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurpose = [
    ([0.0, 0.0],    [0.0]), # [0, 0] => 0
    ([0.0, 1.0],    [1.0]), # [0, 1] => 1
    ([1.0, 0.0],    [1.0]), # [1, 1] => 1
    ([1.0, 1.0],    [0.0])  # [1, 0] => 0
]

thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurposeneuralnets = NeuralNet(2, 1, 1) # this changes based on question parameters
thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurposeneuralnets.train(thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurpose, 0.5 , .1, 5000, 100)

print(thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurposeneuralnets.get_ih_weights())
print()
print(thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurposeneuralnets.get_ho_weights())

print(thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurposeneuralnets.test_with_expected(thisistherealtrainingdatathatiusedinthisprojectforXORpurposesuponpurpose))

"""
thisissomerealtrainingdataforthepurposeofpoliticalpowers = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0]),
]

potlicol = NeuralNet(5, 6, 1)
potlicol.train(thisissomerealtrainingdataforthepurposeofpoliticalpowers)\

print(potlicol.evaluate([1, 1, 1, .1, .1]))
print(potlicol.evaluate([.5, .2, .1, .7, .7]))
print(potlicol.evaluate([.8, .3, .3, .3, .8]))
print(potlicol.evaluate([.8, .3, .3, .8, .3]))
print(potlicol.evaluate([.9, .8, .8, .3, .6]))

