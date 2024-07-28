import torch
from copy import copy


rtk_classnames = {
    0: 'background', 1: 'roadAsphalt', 2: 'roadPaved', 3: 'roadUnpaved', 4: 'roadMarking',
    5: 'speedBump', 6: 'catsEye', 7: 'stormDrain', 8: 'manholeCover', 9: 'patch',
    10: 'waterPuddle', 11: 'pothole', 12: 'crack'
}

rtk_sign_labels = torch.Tensor([4, 5, 6])

mocamba_classnames = {
    0: 'road-objects', 1: 'Animals', 2: 'Asphalt', 3: 'Cat-s-Eye', 4: 'Cracks', 5: 'Ego',
    6: 'Hard-Sand', 7: 'Markings', 8: 'Obstacle', 9: 'People', 10: 'Pothole', 11: 'Retaining-wall',
    12: 'Soft-Sand', 13: 'Unpaved', 14: 'Vehicles', 15: 'Wet-sand'
}

shift = len(mocamba_classnames) - 1
rtk2mocamba = torch.Tensor([
    0, 2, shift+1, 13, 7, shift+2, 3, shift+3, shift+4, shift+5, shift+6, shift+7, 4]).long()

rtk2mocamba_classnames = {
    shift+1: 'roadPaved', shift+2: 'speedBump', shift+3: 'stormDrain', shift+4: 'manholeCover', shift+5: 'patch',
    shift+6: 'waterPuddle', shift+7: 'pothole'
}


class IDs:
    def __init__(self, ds_name='mocamba'):
        self.names_possible = ['mocamba', 'mocamba+rtk']
        assert ds_name in self.names_possible, 'Dataset name not recognized'

        self.id2label = copy(mocamba_classnames)
        if ds_name == 'mocamba+rtk':
            self.id2label.update(rtk2mocamba_classnames)

        self.n_classes = len(self.id2label)
