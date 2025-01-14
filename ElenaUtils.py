import torch

def vectorize(initial_states):
    out = []
    for d in initial_states:
        out.append([[d['mRNA'], d['P']]])
    return torch.tensor(out).float()

def avg_martingale(model, initial_states, Y):
    initial_conditions = vectorize(initial_states).to(Y)
    results = None
    for ic in initial_conditions:
        if results is None:
            results = model.eval_baseline(0., ic, Y)[0]
        else:
            results += model.eval_baseline(0., ic, Y)[0]

    return results/len(initial_conditions)

