from ecdsa import SigningKey, VerifyingKey, der
from ecdsa.ecdsa import Signature
from ecdsa.numbertheory import inverse_mod
from ecdsa.util import sigdecode_string
import hashlib


def compute_d(curve, r1, s1, h1, r2, s2, h2, hashfunc):

    # Extract order from curve
    order = curve.order

    """ TODO: compute the secret exponent d that can then be verified using the code below """

    # We calculate the inverse of the ephemeral key first
    k_inv = (s1 - s2)*(pow(h1 - h2, -1 ,order))
    
    # Then we calculate the real value of k
    k = pow(k_inv,-1,curve.order)

    # Then we calculate d
    d = ((s1 * k - h1) * pow(r1, -1, order )) % order
    
    signing_key = SigningKey.from_secret_exponent(d, curve=curve, hashfunc=hashfunc)
    if signing_key.get_verifying_key().pubkey.verifies(h1, Signature(r1, s1)):
        print("Key recovery successful!")
        return signing_key       

    return None    


if __name__ == "__main__":

    # The Public key
    public_key_pem = ''''
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEAJ7Vt1EenCZAYUmIUEq08HBfVqwL
8o4GPCHcP6JzVQM8pxDMna9KVuRP/DABLNWmBXnOPys7KrIEjuUyb579Fg==
-----END PUBLIC KEY-----
'''

    # Transform PEM public key to python VerifyingKey type
    public_verification_key = VerifyingKey.from_pem(public_key_pem.strip())

    # Signatures
    sig_1 = b'\xc0\xdd$\x1aP\xd4\x8f\x99\xfc\xc7\xa1\x86\xa6\xd4N\x07c\xec\x90G\x8e\x1d\xef\x8e6\xf5\xc4\xe9P\xd6z\xfbm\xf1\xc4\x86\xbb\xac\x12\x81F\x89\xdc#\x7fI#{\\\xb4\xba\xf3\xcf\xb2\x8d\xbd\xaa\xb9\xfeG)\x81y\x87'
    sig_2 = b"\xc0\xdd$\x1aP\xd4\x8f\x99\xfc\xc7\xa1\x86\xa6\xd4N\x07c\xec\x90G\x8e\x1d\xef\x8e6\xf5\xc4\xe9P\xd6z\xfb\xb7\xf91\xf8\xa8\xbd\xd7\x84\x98\xc5w\x0b=\xdc\xe1\xd7\n5\xe7NR'\xef\xa6+\xe8\x04\xce\xf4\x93\x82i"

    r1, s1 = sigdecode_string(sig_1, public_verification_key.pubkey.order)
    r2, s2 = sigdecode_string(sig_2, public_verification_key.pubkey.order)

    # Hashed messages
    msg_1_int_hash = 67695512670386678062547071442295898484089885666034394233603385968388380454424
    msg_2_int_hash = 109597007309208863617441535689493244577862790341600642998103435829172881609118


    # Launch exploit to try to get private key
    private_key = compute_d(public_verification_key.curve, r1, s1, msg_1_int_hash, r2, s2, msg_2_int_hash, hashlib.sha256)

    # Print the recovered private key
    if private_key:
        print("Private key is:\n{}".format(private_key.to_pem()) )
        # Decimal version to be used with openssl
        print(int(private_key.privkey.secret_multiplier))
        with open("private-key.txt", "w") as f:
            f.write(str(int(private_key.privkey.secret_multiplier)) + "\n")

    else:
        print("No private key found :-(")
