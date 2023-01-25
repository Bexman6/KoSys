import numpy as np
from numpy import linalg as la
import copy

'''LLL input basis used in the LLL example'''
#basis = np.array([[19,2,32,46,3,33],[15,42,11,0,3,24],[43,15,0,24,4,16],[20,44,44,0,18,15],[0,48,35,16,31,31],[48,33,32,9,1,29]]).astype(float)
'''LLL input basis used in the exercise'''
basis = np.array([[20,51,35,59,73,73],[14,48,33,61,47,83],[95,41,48,84,30,45],[0,42,74,79,20,21],[6,41,49,11,70,67],[23,36,6,1,46,4]]).astype(float)
'''LLL orthogonalised basis'''
orthobasis = basis.copy()


'''LLL parameter delta'''
#DELTA = 0.75
DELTA = 0.99

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
    
    # We will split it in different parts

    # Calculating the determinant
    det = abs(get_determinant(basis))

    # Produce the norm of the basis vectors
    normed_vec = [la.norm(np.array(vec)) for vec in basis]

    # Calculate the product of the normed basis vecs
    prod = np.product(normed_vec)

    # Calculate the result in the inner bracket
    temp = det/prod

    # Calculate the last step
    result = pow(temp,1/len(normed_vec))

    # More closer to the one, the better 
    print(result)

def lll_reduction():
    '''Compute LLL-reduced basis.'''

    # First we need to create the orthogonal basis through Gram-Schmidt
    gram_schmidt()

    # Run Variables to count the swaps and steps
    steps = 0
    swaps = 0

    # TODO: do LLL-reduction
    
    k = 1
    # Get the amount of basisvectors
    n = basis[0].size

    # Not less equal here
    while k < n:
        
        # Reduction step
        for j in range(k-1, -1, -1):
            basis[k] -= np.round(mu(orthobasis[j], basis[k])) * basis[j]

        # Recreate the orthobasis
        gram_schmidt()

        # Check the Lovasz condition
        if la.norm(orthobasis[k])**2 >= (DELTA - mu(orthobasis[k-1], basis[k])**2) * la.norm(orthobasis[k-1])**2:
            k += 1
        else:
            swaps += 1
            v = copy.copy(basis[k-1])
            basis[k-1] = basis[k]
            basis[k] = v
            k = max(k-1, 1)

        # Recreate the orthobasis another time
        gram_schmidt()

    #print_step(steps, "some place", basis) ; steps += 1

    '''Output'''
    # TODO: adjust after implementation if needed
    print('Number of Swaps:\n', swaps)
    print('LLL Reduced Basis:\n', basis)
    get_hadamard_ratio(basis)
    get_shortest_vector(basis)

def main():

    '''examples'''
    example_LLL   = 0
    example_LLL_short = 0
    hadamard_only = 0
    exercise_LLL = 1
    
    global basis
    global orthobasis
    global DELTA

    if hadamard_only:
        # Basis a
        bas_1 = np.array([[213, 312],[-437, 105]])

        # Basis b
        bas_2 = np.array([[2937,11223], [-1555, -5888]])

        # Calculate the hadamard ratio of basis 1
        print("Hadamard ration of Basis 1:")
        get_hadamard_ratio(bas_1)

        # Calculate Basiswechselmatrix with the gauss algo
        basis_wech_matrix = la.solve(bas_1,bas_2)
        print("\nBasiswechselmatrix of base1 and base2:")
        print(basis_wech_matrix)

        # Calculate the determinant of the Basiswechselmatrix
        print("\nDeterminat of Basiswechselmatrix:")
        print(get_determinant(basis_wech_matrix))

    	# Calculate the hadamard ratio of basis 2
        print("\nHadamard ration of Basis 2:")
        get_hadamard_ratio(bas_2)

    if exercise_LLL:

        # For exercise 5.4 we have to reverse the order of the elements
        basis = np.flipud(basis)

        # As the other examples the orthobasis needs to be initialized
        orthobasis = basis.copy()

        # Calculate determinant and ratio
        print('Determinant:\n', get_determinant(basis))
        get_hadamard_ratio(basis)
        get_shortest_vector(basis)
        print('----------------------')
        
        lll_reduction()

        print('----------------------')
        #get_hadamard_ratio(basis)



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