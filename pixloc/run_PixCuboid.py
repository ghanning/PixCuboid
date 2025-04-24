import argparse
import json
import time
from pathlib import Path

import torch
import tqdm
from omegaconf import OmegaConf

from .pixlib.datasets import get_dataset
from .pixlib.models import get_model
from .pixlib.utils.experiments import get_best_checkpoint
from .pixlib.utils.tensor import batch_to_device
from .pixlib.utils.tools import AverageMetric, set_seed


def main(conf, experiment: str, split: str, output_path: Path) -> None:
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(split, shuffle=False)

    best_cp = get_best_checkpoint(experiment)
    best_cp = torch.load(best_cp, map_location='cpu', weights_only=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_model(conf.model.name)(conf.model).to(device)
    model.load_state_dict(best_cp['model'])
    model.eval()

    start_time = time.time()
    cuboid_preds = []
    success_metric = AverageMetric()

    for data in tqdm.tqdm(loader):
        data = batch_to_device(data, device, non_blocking=True)

        with torch.no_grad():
            pred = model(data)

            if 'points3D' in data:
                loss = model.loss(pred, data)
                success_metric.update(loss['success'])

        for cuboid_opt in pred['cuboid_opt'][-1]:
            cuboid_pred = {x: getattr(cuboid_opt, x).cpu().numpy().tolist() for x in ('R', 't', 's')}
            cuboid_preds.append(cuboid_pred)

    elapsed = time.time() - start_time
    num_pred = len(loader.dataset)
    print(f'Predicted {num_pred} layouts in {elapsed:.2f} s.')
    print(f'Average time per prediction: {1000.0*elapsed/num_pred:.2f} ms')

    print(f'Success: {100.0*success_metric.compute():.2f}%')

    with open(output_path, 'w') as f:
        json.dump(cuboid_preds, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PixCuboid')
    parser.add_argument('--experiment', '-exp', required=True, help='Experiment to use')
    parser.add_argument('--conf', '-c', type=Path, required=True, help='Path to config file')
    parser.add_argument('--split', '-s', required=True, choices=('train', 'val', 'test'), help='Data split')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Path to output JSON prediction file')
    parser.add_argument('extra_args', nargs='*')
    parser.add_argument('--seed', '-seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    conf = OmegaConf.load(args.conf)
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.extra_args))
    main(conf, args.experiment, args.split, args.output)
