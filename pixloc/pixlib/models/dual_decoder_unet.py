import torch
import torch.nn as nn

from .unet import AdaptationBlock, DecoderBlock, UNet
from .utils import checkpointed


class DualDecoderUNet(UNet):
    default_conf = {
        'compute_edge_map': False,
        'compute_edge_uncertainty': False,
    }

    def build_decoder(self, conf, skip_dims, output_dim, compute_uncertainty):
        # Decoder
        if conf.decoder is not None:
            assert len(conf.decoder) == (len(skip_dims) - 1)
            Block = checkpointed(DecoderBlock, do=conf.checkpointed,
                                 use_reentrant=False)
            norm = eval(conf.decoder_norm) if conf.decoder_norm else None

            previous = skip_dims[-1]
            decoder = []
            for out, skip in zip(conf.decoder, skip_dims[:-1][::-1]):
                decoder.append(Block(previous, skip, out, norm=norm))
                previous = out
            decoder = nn.ModuleList(decoder)
        else:
            decoder = None

        # Adaptation layers
        adaptation = []
        if compute_uncertainty:
            uncertainty = []
        for idx, i in enumerate(conf.output_scales):
            if conf.decoder is None or i == (len(self.encoder) - 1):
                input_ = skip_dims[i]
            else:
                input_ = conf.decoder[-1 - i]

            # out_dim can be an int (same for all scales) or a list (per scale)
            dim = output_dim
            if not isinstance(dim, int):
                dim = dim[idx]

            block = AdaptationBlock(input_, dim)
            adaptation.append(block)
            if compute_uncertainty:
                uncertainty.append(AdaptationBlock(input_, 1))
        adaptation = nn.ModuleList(adaptation)
        if conf.compute_uncertainty:
            uncertainty = nn.ModuleList(uncertainty)
        else:
            uncertainty = None

        return decoder, adaptation, uncertainty

    def _init(self, conf):
        # Encoder
        self.encoder, skip_dims = self.build_encoder(conf)

        # Decoder(s)
        self.decoder, self.adaptation, self.uncertainty = self.build_decoder(
            conf, skip_dims, conf.output_dim, conf.compute_uncertainty
        )

        if conf.compute_edge_map:
            self.edge_decoder, self.edge_adaptation, self.edge_uncertainty = (
                self.build_decoder(
                    conf, skip_dims, 1, conf.compute_edge_uncertainty
                )
            )

        self.scales = [2**s for s in conf.output_scales]

    def decode(self, decoder, adaptation, uncertainty, skip_features):
        if decoder is not None:
            pre_features = [skip_features[-1]]
            for block, skip in zip(decoder, skip_features[:-1][::-1]):
                pre_features.append(block(pre_features[-1], skip))
            pre_features = pre_features[::-1]  # fine to coarse
        else:
            pre_features = skip_features

        out_features = []
        for adapt, i in zip(adaptation, self.conf.output_scales):
            out_features.append(adapt(pre_features[i]))

        if uncertainty is not None:
            confidences = []
            for layer, i in zip(uncertainty, self.conf.output_scales):
                unc = layer(pre_features[i])
                conf = torch.sigmoid(-unc)
                confidences.append(conf)
        else:
            confidences = None

        return out_features, confidences

    def _forward(self, data):
        image = data['image']

        if image.ndim == 5:
            pred = [self._forward({'image': i}) for i in image]
            pred = {
                k: [torch.stack([p[k][i] for p in pred]) for i in range(len(v))]
                for k, v in pred[0].items()
                if v is not None
            }
            return pred

        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]

        skip_features = []
        features = image
        for block in self.encoder:
            features = block(features)
            skip_features.append(features)

        pred = {}
        pred['feature_maps'], pred['confidences'] = self.decode(
            self.decoder, self.adaptation, self.uncertainty, skip_features
        )

        if self.conf.compute_edge_map:
            pred['edge_maps'], pred['edge_confidences'] = self.decode(
                self.edge_decoder,
                self.edge_adaptation,
                self.edge_uncertainty,
                skip_features,
            )
            pred['edge_maps'] = [torch.sigmoid(-x) for x in pred['edge_maps']]

        return pred
