# ------------------------------------------------------------------------------------ distance metrics
# L2
def simpleL2(a,b):
    q = a-b
    return np.sqrt((q*q).sum())

#Weighted Euclidean Distance
def weightedL2(a,b,w):
    q = a-b
    return np.sqrt((w*q*q).sum())

#  normalized Euclidean distance
def normalizedL2(a,b,s):
    q = a-b
    return np.sqrt((q*q/s*s).sum())

# L1 family
# L1
def simpleL1(a,b):
    q = a-b
    return np.absolute(q).sum()

# absolute Weighted L1
def absoluteWxL1(a,b,w):
    q = a-b
    return np.absolute(w*q).sum()

# absolute normalized Weighted L1    
def absoluteWxL1_N(a,b,w):
    q = a-b
    Cw = np.absolute(w).sum()
    return np.absolute(w*q).sum()/Cw

#semantic value
def semantic_v(w,f,b):
     return (w*f).sum() + b

#semantic value normalized    
def semantic_vN(w,f,b):
     Cw = np.absolute(w).sum()
     return ((w*f).sum() + b)/Cw 
    
#semantic absolute value
def semantic_av(w,f,b):
     return (np.absolute(w)*f).sum() + b    
    
#semantic absolute value normalized
def semantic_avN(w,f,b):
     Cw = np.absolute(w).sum()
     return (np.absolute(w)*f).sum()/Cw   