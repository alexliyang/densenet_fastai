from densenet import *
import torchvision
from collections import OrderedDict

dn_models = {
    'densenet121': densenet121, 
    'densenet169': densenet169, 
    'densenet201': densenet201, 
    'densenet161': densenet161,
}

torch_models = {
    'densenet121': torchvision.models.densenet121, 
    'densenet169': torchvision.models.densenet169, 
    'densenet201': torchvision.models.densenet201, 
    'densenet161': torchvision.models.densenet161,
}

for m in tqdm(dn_models.keys()):
    print(f"Fixing {m}")
    # densenet with layer names fixed
    dnetm = models[m]()
    # original densenet 
    dnet = torch_models[m](True).eval()

    # get the state dict of 
    dnet_sdict = dnet.state_dict()
    d_keys = dnet_sdict.keys()
    dm_keys = dnetm.state_dict().keys() # modified densenet keys

    dnetm.load_state_dict(OrderedDict(zip(dm_keys, dnet_sdict.values())))
    dnetm.eval()
    dnetm_sdict = dnetm.state_dict()

    for k1, k2 in zip(d_keys, dm_keys):
        assert torch.equal(dnet_sdict[k1], dnetm_sdict[k2]), f"{k1}!={k2}"

    torch.save(dnetm.state_dict(), model_locs[m])
    print(f"Saving to {model_locs[m]}\n")
    
print("Done!")
