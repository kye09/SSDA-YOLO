import argparse
import yaml
from pathlib import Path


class WeightEMA:
    """Exponential moving average for updating teacher model."""

    def __init__(self, teacher, student, alpha=0.999):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        for p in self.teacher.parameters():
            p.requires_grad = False

    def update(self):
        import torch

        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), self.student.parameters()):
                t.data.mul_(self.alpha).add_(s.data * (1.0 - self.alpha))


def parse_data_yaml(data_yaml):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    root = Path(data.get('path', '.'))

    def gather(key):
        return [str(root / p) for p in data.get(key, [])]

    source = gather('train_source_real') + gather('train_source_fake')
    target = gather('train_target_real') + gather('train_target_fake')
    info = {'nc': data['nc'], 'names': data['names'], 'channels': 3}
    return source, target, info


def build_loaders(source, target, info, cfg):
    from ultralytics.data import build_yolo_dataset, build_dataloader

    dataset_s = build_yolo_dataset(cfg, source, cfg.batch, info, mode='train')
    dataset_t = build_yolo_dataset(cfg, target, cfg.batch, info, mode='train')
    loader_s = build_dataloader(dataset_s, batch=cfg.batch, workers=2, shuffle=True)
    loader_t = build_dataloader(dataset_t, batch=cfg.batch, workers=2, shuffle=True)
    return loader_s, loader_t


def train(opt):
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_cfg({'imgsz': opt.img_size, 'batch': opt.batch_size, 'epochs': opt.epochs, 'lr0': opt.lr, 'task': 'segment'})
    source, target, info = parse_data_yaml(opt.data)
    loader_s, loader_t = build_loaders(source, target, info, cfg)

    student = YOLO(opt.weights).to(device)
    teacher = YOLO(opt.weights).to(device)
    ema = WeightEMA(teacher.model, student.model, opt.ema_alpha)
    optimizer = torch.optim.SGD(student.model.parameters(), lr=opt.lr, momentum=0.9)
    mse = torch.nn.MSELoss()

    for epoch in range(opt.epochs):
        for batch_s, batch_t in zip(loader_s, loader_t):
            for k, v in batch_s.items():
                if hasattr(v, 'to'):
                    batch_s[k] = v.to(device)
            for k, v in batch_t.items():
                if hasattr(v, 'to'):
                    batch_t[k] = v.to(device)

            preds_s = student.model(batch_s['img'])
            loss_s = student.model.loss(batch_s, preds_s)

            with torch.no_grad():
                t_out = teacher.model(batch_t['img'])
            s_out = student.model(batch_t['img'])
            loss_u = mse(s_out[0], t_out[0].detach()) if isinstance(s_out, (list, tuple)) else mse(s_out, t_out.detach())

            loss = loss_s + opt.lambda_u * loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

        print(f'Epoch {epoch + 1}/{opt.epochs}: loss={loss.item():.4f}')

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    student.model.save(save_dir / 'student.pt')
    teacher.model.save(save_dir / 'teacher.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='dataset yaml')
    parser.add_argument('--weights', type=str, default='yolov8n-seg.pt', help='pretrained weights')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lambda-u', type=float, default=1.0)
    parser.add_argument('--ema-alpha', type=float, default=0.999)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='ssda_yolov8_seg')
    parser.add_argument('--hyp', type=str, default=None, help='hyperparameter yaml')
    opt = parser.parse_args()

    if opt.hyp:
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)
        for k, v in hyp.items():
            attr = k.replace('-', '_')
            if hasattr(opt, attr):
                setattr(opt, attr, v)

    train(opt)
