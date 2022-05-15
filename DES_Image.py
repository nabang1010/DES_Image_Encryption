import cv2
import numpy as np
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]
FP = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]
EBox = [32,1,2,3,4,5,
        4,5,6,7,8,9,
        8,9,10,11,12,13,
        12,13,14,15,16,17,
        16,17,18,19,20,21,
        20,21,22,23,24,25,
        24,25,26,27,28,29,
        28,29,30,31,32,1]
parity = [57,49,41,33,25,17,9,1,
          58,50,42,34,26,18,10,2,
          59,51,43,35,27,19,11,3,
          60,52,44,36,63,55,47,39,
          31,23,15,7,62,54,46,38,
          30,22,14,6,61,53,45,37,
          29,21,13,5,28,20,12,4]
SBox =[	[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
	 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
	 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
	 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
	[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
	 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
	 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
	 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
	[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
	 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
	 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
	 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
	[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
	 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
	 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
	 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
	[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
	 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
	 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
	 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
	[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
	 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
	 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
	 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
	[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
	 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
	 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
	 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
	[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
	 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
	 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
	 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],]
F_PBox = [16, 7, 20, 21, 29, 12, 28, 17,
          1, 15, 23, 26, 5, 18, 31, 10,
          2, 8, 24, 14, 32, 27, 3, 9,
          19, 13, 30, 6, 22, 11, 4, 25 ]
key_PBox = [14,17,11,24, 1,  5,
             3,28,15, 6,21,10,
            23,19,12, 4,26, 8,
            16, 7,27,20,13, 2,
            41,52,31,37,47,55,
            30,40,51,45,33,48,
            44,49,39,56,34,53,
            46,42,50,36,29,32]
def XOR (A,B):
    result = (np.logical_xor(A,B)).astype(int)
    return result
def permutation (A,B):
    permutation = np.zeros(len(B), dtype=int)
    j = 0
    for i in B:
        permutation[j] = A[i-1]
        j += 1
    return (permutation)
def Expansion_Box(Right):
    Expansion_Box = np.zeros(len(EBox), dtype=int)
    j = 0
    for i in EBox:
        Expansion_Box[j] = Right[i-1]
        j += 1
    return (Expansion_Box)   
def Sbox (A):
    S_Box = ""
    k = []
    t = 0
    m = 0
    for i in range(1,len(A)+1):
        k.append(A[i-1])
        if i % 6 == 0:
            t = k[0]*2+k[5]*1
            m = k[1]*8+k[2]*4+k[3]*2+k[4]*1
            x = t*16 + m
            S_Box += '{0:04b}'.format((SBox[(i//6)-1][x]))     
            t = m = 0
            k =[]
    return S_Box  
def Straight_Box(A):
    Straight_Box = np.zeros(len(F_PBox), dtype=int)
    j = 0
    for i in F_PBox:
        Straight_Box[j] = A[i-1]
        j += 1
    return (Straight_Box)
def Compression (A,B):
    C = np.concatenate((A, B), axis = None)
    Compression = np.zeros(len(key_PBox), dtype = int)
    j = 0
    for i in key_PBox:
        Compression[j] = C[i-1]
        j += 1
    return (Compression)
def parity_Key(key):
    parity_Key = np.zeros(len(parity), dtype=int)
    j = 0
    for i in parity :
        parity_Key[j] = key[i-1]
        j += 1
    return (parity_Key)
def keyshift (A,n):
    if (n == 1) or (n == 2) or (n == 9) or (n == 16):
        A= np.roll(A,-1)
        return A
    else:
        A = np.roll(A, -2)
        return A
def to16key (key56,n):
    to16key = ""
    A = key56[:28]
    B = key56[28:]
    for i in range(1,n+1):
        A = keyshift(A,i)
        B = keyshift(B,i)
    to16key = Compression(A,B)
    return (to16key)
def function_DES (Right, key56 ,n):
    expansion = Expansion_Box(Right)
    key = to16key (key56,n)
    t = XOR(expansion,key)
    s = Sbox (t)
    output = Straight_Box(s)
    return (output)
def Round (Left, Right, key56, n):
    t = function_DES(Right, key56, n)
    R = XOR(Left, t)
    L = Right
    return (L,R)
def encryption_z (plaintext, key):
    if (len(plaintext)%8)==0:
        index = (len(plaintext)//8)
    else:
        index = (len(plaintext)//8)+1
    plaintext_bin =""
    key_bin=""
    for t in range (index*8):
        k=""
        ke=""
        if (t >= len(plaintext)) and (t < 8*index):
            k = '{0:08b}'.format((0))
            plaintext_bin += k    
        else:
            k = '{0:08b}'.format((ord((plaintext[t]))))
            plaintext_bin += k
        if t < 8:
            ke = '{0:08b}'.format((ord((key[t]))))
            key_bin+=ke
    cyphertext =""
    key56 = parity_Key(key_bin)
    for i in range(index):
        plaintext_bin_1 = ""
        for j in range (64):
            plaintext_bin_1 += (plaintext_bin[j + i*64])
        plaintext_bin_1 = permutation (plaintext_bin_1,IP)
        A = plaintext_bin_1[:32]
        B = plaintext_bin_1[32:]
        for k in range(1,17):
            A, B = Round (A, B, key56, k)
        ciphertext_bin8 = np.concatenate((B, A), axis = None)
        ciphertext_bin8 = permutation (ciphertext_bin8,FP)
        x = []
        y = 0
        for l in range(1,65):
            x.append(ciphertext_bin8[l-1])
            if l%8 == 0:
                y = x[0]*128+x[1]*64+x[2]*32+x[3]*16+x[4]*8+x[5]*4+x[6]*2+x[7]*1
                cyphertext += chr(y)
                x = []
                y = 0
    return (cyphertext)
def key_to_bin(key):
    key_bin = ""
    for i in key:
        key_bin += '{0:08b}'.format((ord((i))))
    return key_bin
def encryption (plain_text_bin, key):
    key_bin = key_to_bin(key)
    key56 = parity_Key(key_bin)
    plaintext_bin_1 = permutation (plain_text_bin,IP)
    
    A = plaintext_bin_1[:32]
    B = plaintext_bin_1[32:]
    for i in range(16):
        A, B = Round (A, B, key56, i+1)
    ciphertext_bin8 = np.concatenate((B, A), axis = None)
    ciphertext_bin8 = permutation (ciphertext_bin8,FP)
    return ciphertext_bin8
def dec_to_bin(num):    
    bnr = bin(num).replace('0b','')
    x = bnr[::-1] 
    while len(x) < 8:
        x += '0'
    bnr = x[::-1]
    return bnr
def des_image_encryption(image,key):
    h,w = image.shape
    image_1d  = np.reshape(image,(1, h*w))
    plain_text = ""
    cipher_text = []
    for i in range(h*w):
        plain_text += dec_to_bin(image_1d[0][i])
        if ((i+1)%8==0)and (i!=0):
            cipher_text.append(encryption(plain_text,key))
            plain_text = ""
    res = (h*w)%8
    res_plain_text = ""
    for i in range((h*w)-res, (h*w)):
        res_plain_text +=dec_to_bin(image_1d[0][i]) 
    for i in range((64-len(res_plain_text))//8):
        res_plain_text +=dec_to_bin(0) 
    cipher_text.append(encryption(res_plain_text,key))
    return cipher_text
def decryption (cipher_text_bin, key):
    key_bin = key_to_bin(key)
    key56 = parity_Key(key_bin)
    cipher_text_1 = permutation (cipher_text_bin,IP)
    A = cipher_text_1[:32]
    B = cipher_text_1[32:]
    for i in range( 16, 0, -1):
        A, B = Round (A, B, key56, i)
    received_text_bin8 = np.concatenate((B, A), axis = None)
    received_text_bin8 = permutation (received_text_bin8,FP)
    return received_text_bin8
def des_image_decryption(cipher_image,key):
    h,w = cipher_image.shape
    image_1d  = np.reshape(cipher_image,(1, h*w))
    cipher_text = ""
    received_text = []
    for i in range(h*w):
        cipher_text += dec_to_bin(image_1d[0][i])
        if ((i+1)%8==0)and (i!=0):
            received_text.append(decryption(cipher_text,key))
            cipher_text = ""
    res = (h*w)%8
    res_cipher_text = ""
    for i in range((h*w)-res, (h*w)):
        print("++",i)
        res_cipher_text +=dec_to_bin(image_1d[0][i]) 
    for i in range((64-len(res_cipher_text))//8):
        res_cipher_text +=dec_to_bin(0) 
    received_text.append(decryption(res_cipher_text,key))
    return received_text
def main():
    img = cv2.imread('test.png')
    key = "ltmm1234"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image',img)
    cv2.waitKey(1)
    cv2.imshow('Gray',gray)
    cv2.waitKey(1)
    cipher_text = des_image_encryption(gray,key)
    cipher_image_org = []
    for i in cipher_text:
        pixel = ""
        for j in range(len(i)):
            pixel += str(i[j])
            if (j+1)%8==0:
                cipher_image_org.append(int(pixel,2))
                pixel =""
    h,w = gray.shape
    cipher_image = np.reshape(cipher_image_org[:(h*w)],(h,w))
    cv2.imshow('Image Encryption',cipher_image)
    cv2.waitKey(1)
    received_text_de = des_image_decryption(cipher_image,key)
    received_text_org = []
    for i in received_text_de:
        pixel = ""
        for j in range(len(i)):
            pixel += str(i[j])
            if (j+1)%8==0:
                received_text_org.append(int(pixel,2))
                pixel =""
    received_image = np.reshape(received_text_org[:(h*w)],(h,w))
    cv2.imshow('Image Decryption',received_image)
    cv2.waitKey(1)
main()
