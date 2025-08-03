from utils import PINN, QRes, FLS, PINNsFormer, PINNMamba, piratenet

def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'PINNsFormer': PINNsFormer,
        'PINNMamba': PINNMamba,
        'PirateNet':piratenet,
    }
    return model_dict[args.model]