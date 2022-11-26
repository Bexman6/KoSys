from polynomials import *
from modules import *

def keygen():

    """
    If you build a vector v of polynomials p0 and p1, you can generate the polynomials e.g., by using
    p0 = R([0, 1, 1, 0]) and p1 = R([-1, 0, 1, 0]).

    If you combine them in a matrix X with X = M([p0, p1]), it's a row vector: 
    [5 + 4*x + 10*x^2 + 6*x^3, 7 + 10*x + 16*x^2 + x^3]
    After transposing it, it's a column vector:
    [5 + 4*x + 10*x^2 + 6*x^3]
    [ 7 + 10*x + 16*x^2 + x^3]
    """

    # Generate a secret key which is a vector of elements pulled from a centered polynomial distribution
    s0 = R([1,0,0,-1])
    s1 = R([0,-1,1,0])
    
    s = M([s0, s1]).transpose()
    print("Secret key s")
    print(s)

    # Generate a 2x2 matrix with elements taken randomly from R_q
    A00 = R([8,0,1,16])
    A01 = R([1,9,9,2])
    A10 = R([13,10,3,5])
    A11 = R([15,4,8,6])

    A = M([[A00, A01], [A10, A11]])
    print("Public key (A)")
    print(A)

    # Generate random error vector from binomial distribution
    e0 = R([0,1,0,-1])
    e1 = R([1,0,1,0])

    e = M([e0, e1]).transpose()
    print("Error vector e")
    print(e)

    # Compute t
    t = A @ s + e
    print("Public key (t)")
    print(t)

    return (A, t), s

def enc(m, public_key):
    """
    TODO: encrypt the message with the given public key and the fixed parameters from the exercise sheet
    The comments give you hints about already implemented functions that can be used.
    """

    # Generate random vector r from binomial distribution
    r = None

    # Generate random vector e_1 from binomial distribution
    e_1 = None

    # Generate random polynomial e_2 from binomial distribution
    e_2 = None

    A, t = public_key

    # R.decode(msg) takes the msg byte array and turns it into a polynomial
    # Decompression is then used to create error tolerance gaps by sending the 
    #  message bit 0 to 0 and bit 1 to rounded q/2
    

    # Matrix multiplication can be done with '@', see keygen 

    u = None

    # To get a poylomial out of a vector v of polynomials of dimension 1x1, use (v)[0][0]
    v = None

    return u, v

def dec(u, v, s):

    """
    TODO: decrypt the message with the given secret key s
    The comments give you hints about already implemented functions that can be used.
    """

    # To get a poylomial out of a vector v of polynomials of dimension 1x1, use (v)[0][0]


    # Compression (via compress(1)) is used to decrypt to a 1 if m_n is closer to rounded q/2 than to 0
    #  and decrypt to 0 otherwise


    # msg_pol.encode(l=2) takes a vector of polynomials and converts the polynomials to byte arrays
    #  that are then concatenated
    # The parameter is the length (32*l) of the byte array


    return None

if __name__ == '__main__':

    """
    This is a simplified implementation of a kyber keygen, enc and dec.
    All parameters are smaller to make it more readable.

    Polynomial ring (R) and the vector space, where the field of scalars is replaced with a ring (M)
    This means that q = 17 and f = x^4 + 1
    """
    
    R = PolynomialRing(17, 4) # R = GF(q) / (X^n + 1)
    M = Module(R)

    # Message to encode (example-msg is 'A')
    m = bytes([65])
    print("Message to encrypt: ", m)
    print(R.decode(m))

    # Generate keypair
    # pub is a tuple, priv is a single vector
    pub, priv = keygen()

    # Encrypt message
    u, v = enc(m, pub)

    # Decrypt message
    m_dec = dec(u, v, priv)
    print("Decrypted message: ", m_dec)