
import matplotlib.pyplot as plt
import matplotlib

# Setup for latex rendering
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Computer Modern font for academic look

def save_formula(formula, filename, fontsize=20):
    fig = plt.figure(figsize=(6, 2))
    text = fig.text(0.5, 0.5, f"${formula}$", 
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=fontsize)
    
    # Remove axes and background
    plt.axis('off')
    
    # Save
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300, transparent=True)
    plt.close()
    print(f"Saved {filename}")

# Pmax formula
pmax_eq = r"P_{\max} = \max_c p(y=c|x)"
save_formula(pmax_eq, "pmax_formula.png", fontsize=28)

# SAL formula
sal_eq = r"SAL = \cos(f(x), \mu_{\hat{y}}) = \frac{f(x) \cdot \mu_{\hat{y}}}{\|f(x)\| \|\mu_{\hat{y}}\|}"
save_formula(sal_eq, "sal_formula.png", fontsize=28)
