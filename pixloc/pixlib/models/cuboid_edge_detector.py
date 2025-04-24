import torch
from torchvision.transforms import InterpolationMode, Resize

from . import get_model
from .two_view_refiner import TwoViewRefiner


class CuboidEdgeDetector(TwoViewRefiner):
    default_conf = {
        'extractor': {
            'name': 'dual_decoder_unet',
        },
        'loss': 'MSELoss',
        'weight_edge': None,
    }
    required_data_keys = ['image']

    def _init(self, conf):
        super()._init(conf)
        self.extractor = get_model(conf.extractor.name)(conf.extractor)
        self.loss_fn = eval('torch.nn.' + conf.loss)(reduction='none')
        self.loss_name = conf.loss.lower().replace('loss', '')

    def _forward(self, data):
        image = data['image']
        assert image.shape[1] == 1
        pred = self.extractor({'image': image[:, 0]})
        return pred

    def loss(self, pred, data):
        edge_image = data['edge_image']
        assert edge_image.shape[1] == 1
        edge_image = edge_image[:, 0]

        losses = {'total': 0.}
        for i in reversed(range(len(self.extractor.scales))):
            edge_image_pred = pred['edge_maps'][i]

            if edge_image_pred.shape == edge_image.shape:
                edge_image_target = edge_image
            else:
                resize = Resize(
                    edge_image_pred.shape[2:], InterpolationMode.BICUBIC)
                edge_image_target = resize(edge_image)

            err = self.loss_fn(edge_image_pred, edge_image_target)
            if self.conf.weight_edge:
                weight_edge = self.conf.weight_edge
                err *= weight_edge + edge_image_target * (1. - weight_edge)
            loss = err.flatten(start_dim=1).mean(-1)

            losses[f'{self.loss_name}/{i}'] = loss
            losses['total'] += loss

        return losses

    def metrics(self, pred, data):
        return {}
