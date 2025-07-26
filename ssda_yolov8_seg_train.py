import argparse
import yaml
from pathlib import Path

import torch
from ultralytics import YOLO


class WeightEMA:
    """Exponential moving average of model weights for teacher update."""
    def __init__(self, teacher, student, alpha=0.999):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self):
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(self.alpha).add_(s.data * (1.0 - self.alpha))


def load_dataset(yaml_file, split):
    """Load dataset path list from YAML."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)
    imgs = []
    for path in data[split]:
        imgs += list(Path(path).glob('*.jpg'))
    return imgs


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, labels


def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = YOLO(opt.weights).to(device)
    teacher = YOLO(opt.weights).to(device)
    ema = WeightEMA(teacher.model, student.model, opt.ema_alpha)

    # dataloaders
    source_imgs = load_dataset(opt.data, 'train_source_real')
    target_imgs = load_dataset(opt.data, 'train_target_real')
    dataset_s = student.Dataset(source_imgs, imgsz=opt.img_size)
    dataset_t = student.Dataset(target_imgs, imgsz=opt.img_size, label=False)

    loader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opt.batch_size,
                                           shuffle=True, collate_fn=collate_fn)
    loader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opt.batch_size,
                                           shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.SGD(student.model.parameters(), lr=opt.lr, momentum=0.9)

    for epoch in range(opt.epochs):
        for (imgs_s, labels_s), (imgs_t, _) in zip(loader_s, loader_t):
            imgs_s, imgs_t = imgs_s.to(device), imgs_t.to(device)

            # supervised loss on source
            preds_s = student.model(imgs_s)
            loss_s = student.loss(preds_s, labels_s)

            # pseudo labels from teacher
            with torch.no_grad():
                teacher_preds = teacher.model(imgs_t)
            pseudo = [p[p[..., 4] > opt.conf_thres] for p in teacher_preds]
            preds_t = student.model(imgs_t)
            loss_t = student.loss(preds_t, pseudo)

            loss = loss_s + opt.lambda_u * loss_t
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        print(f'Epoch {epoch+1}/{opt.epochs}: loss={loss.item():.4f}')

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
    parser.add_argument('--conf-thres', type=float, default=0.5)
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
