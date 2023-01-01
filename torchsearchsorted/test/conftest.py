import pytest
import torch

devices = {'cpu': torch.device('cpu')}
if torch.cuda.is_available():
    devices['cuda'] = torch.device('cuda:0')
    else torch.device('cpu')


@pytest.fixture(params=devices.values(), ids=devices.keys())
def device(request):
    return request.param
