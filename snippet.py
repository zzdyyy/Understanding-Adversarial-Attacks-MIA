import matplotlib.pyplot as plt

def reg(x): return (x - x.min()) / (x.max() - x.min())

class c:
    pass

args = c()
args.dataset = 'cxr'
args.attack = 'pgd'
args.characteristics = 'kd'
