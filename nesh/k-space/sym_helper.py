'''
sym_helper.py
Helpful functions for symbolic calculations
v. 0.1 (first version made public) 17JAN2023
Michael R. Gustafson II
'''

import sympy as sym

try:
    import IPython.display as disp
    show = disp.display
except:
    show = sym.pretty_print
  
def printout(solution, variables=None, fp=False):
    # Default case: print all variables
    if variables==None:
        variables = tuple(solution.keys())
    # Fixes issue if solving for a single item
    if not isinstance(variables, (list, tuple)):
        variables = [variables]
    for var in variables:
        show(var)
        if fp:
            show(solution[var].evalf(fp))
        else:
            show(solution[var])
        print()

def makesubs(solution, vartosub, valtosub):
    subsol = solution.copy()
    sublist = list(zip(vartosub, valtosub))
    if type(subsol)==list:
        subsol = makedict(subsol)
      
    for var in subsol:
        subsol[var] = subsol[var].subs(sublist)
    return subsol

def makedict(sollist):
    d = {}
    for eqn in sollist:
        d[eqn.lhs] = eqn.rhs
    return d 

def makesubseqnlist(equations, vartosub, valtosub):
    sublist = list(zip(vartosub, valtosub))
    esub = []    
    for eqn in equations:
      esub.append(eqn.subs(sublist))
    return esub