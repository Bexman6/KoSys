import numpy as np
from numpy import linalg as la

'''LLL input basis used in the LLL example'''
basis = np.array([[19,2,32,46,3,33],[15,42,11,0,3,24],[43,15,0,24,4,16],[20,44,44,0,18,15],[0,48,35,16,31,31],[48,33,32,9,1,29]]).astype(float)
'''LLL input basis used in the exercise'''
#basis = np.array([[20,51,35,59,73,73],[14,48,33,61,47,83],[95,41,48,84,30,45],[0,42,74,79,20,21],[6,41,49,11,70,67],[23,36,6,1,46,4]]).astype(float)
'''LLL orthogonalised basis'''
orthobasis = basis.copy()


'''LLL parameter delta'''
DELTA = 0.75

def get_shortest_vector(basis):
    '''Gets the shortest vector out of a basis.'''
    
    norms = [la.norm(np.array(vec)) for vec in basis]
    print('Shortest vector v{0} with length {1}'.format(np.argmin(norms)+1, np.min(norms)))

def get_determinant(basis):
    '''Returns the determinant of a basis.'''

    return int(np.round(la.det(basis)))

def mu(u, v):
    '''Computes <u,v>/<u,u>, which is the scale used in projection.'''

    return np.dot(u, v) / la.norm(u)**2

def gram_schmidt(): 
    '''Computes Gram Schmidt orthoganalization (without normalization) of a basis.'''

    global orthobasis

    orthobasis[0] = basis[0]

    for i in range(1, basis.shape[1]):  
        orthobasis[i] = basis[i]
        for j in range(0, i):
            orthobasis[i] -= np.dot(mu(orthobasis[j], basis[i]), orthobasis[j])

def print_step(steps, func, basis):
    '''Can be used to show all steps of LLL algorithm for debugging.'''

    print('Step ', steps, f'. After the {func}, the basis is\n', basis)

    # press enter for each step
    input("")

def get_hadamard_ratio(basis):
    '''Computes the hadamard ratio for a given basis.'''
    
    # TODO: get the ratio according to the formula in the exercise sheet

    ratio = 0
    print('Hadamard Ratio:\n', ratio)

def lll_reduction():
    '''Compute LLL-reduced basis.'''

    steps = 0
    swaps = 0

    # TODO: do LLL-reduction

    #print_step(steps, "some place", basis) ; steps += 1

    '''Output'''
    # TODO: adjust after implementation if needed
    print('Number of Swaps:\n', swaps)
    print('LLL Reduced Basis:\n', basis)
    get_hadamard_ratio(basis)
    get_shortest_vector(basis)

def main():
	#TODO: change according to the current exercise
    

    '''examples'''
    example_LLL   = 1
    example_LLL_short = 0
    
    global basis
    global orthobasis
    global DELTA

    if example_LLL:
        ''' LLL input basis from "Hinweise zu den Implementierungsaufgaben"'''
        print('\nexample of LLL')
        basis = np.array([[19,2,32,46,3,33],[15,42,11,0,3,24],[43,15,0,24,4,16],[20,44,44,0,18,15],[0,48,35,16,31,31],[48,33,32,9,1,29]]).astype(float)
        orthobasis = basis.copy()

        print('Determinant:\n', get_determinant(basis))
        get_hadamard_ratio(basis)
        get_shortest_vector(basis)
        print('----------------------')

        lll_reduction()

        print('----------------------')

    if example_LLL_short:
        ''' LLL input basis short example'''
        print('\nshort example of LLL')
        basis = np.array([[15,23,11],[46,15,3],[32,1,1]]).astype(float)
        orthobasis = basis.copy()

        print('Determinant:\n', get_determinant(basis))
        get_hadamard_ratio(basis)
        get_shortest_vector(basis)
        print('----------------------')

        lll_reduction() 
    

if __name__ == "__main__":
    main()