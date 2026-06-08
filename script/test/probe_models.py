import sys, torch
sys.path.insert(0, 'src')
from model.pet_torch_model import build_pet_torch
from model.ulcnn_torch_model import build_ulcnn_torch_model
from model.fea_t_torch_model import build_fea_t_model
from model.iqformer_torch_model import build_iqformer_model
from model.mcldnn_torch_model import build_mcldnn_torch_model
from model.amcnet_torch_model import build_amcnet_torch_model

x = torch.randn(4, 2, 128)
builders = [
    ('pet', lambda: build_pet_torch((2, 128), 11)),
    ('ulcnn', lambda: build_ulcnn_torch_model((2, 128), 11)),
    ('fea_t', lambda: build_fea_t_model((2, 128), 11)),
    ('iqformer', lambda: build_iqformer_model((2, 128), 11)),
    ('mcldnn', lambda: build_mcldnn_torch_model((2, 128), 11)),
    ('amcnet', lambda: build_amcnet_torch_model((2, 128), 11)),
]
for name, fn in builders:
    try:
        m = fn(); m.eval()
        with torch.no_grad():
            y = m(x)
        print(f'{name:10s}: out {tuple(y.shape)}  params {sum(p.numel() for p in m.parameters()):,}')
    except Exception as e:
        print(f'{name:10s}: ERR {repr(e)[:220]}')
