import time, os
import torch

from model.tlee import TLEEModel, BaseModel
from dataset.dataset import get_dataloader
from read_config import config


def main():
    print_info()
    model = get_model()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device('cpu')
    # test_on_jetson_nano(model, device)
    ckpt = get_ckpt(config.test_path, device)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    # val_loader = get_dataloader(config.data_val, batch_size=1)
    dataset_name = config.data.name
    if dataset_name == 'hmdb51':
        data_len = 1530
    else:
        data_len = 101
    s_time = test(model, device, data_len)
    avg_s_time = s_time / (data_len - 1)
    print(f"all {data_len} videos cost {s_time:.4f}s, average costs {avg_s_time:.4f}, FPS: {1 * 10 / avg_s_time:.2f}!")

def get_ckpt(path, device):
    ckpt = torch.load(os.path.join('ckpt/', path, 'ckpt.best.pth.tar'), map_location=device)['state_dict']
    return ckpt

def test(model, device, data_len):
    model.eval()
    branch_num = config.model.branch_num
    s = 0
    for itr in range(data_len):
        spatial_exit = [0 for _ in range(10)]
        pred_true = [-1, -1]
        batch = torch.load(f'../{config.data.name}_val/{config.data.name}_val_{itr}.pth')
        x, y = batch['x'], batch['y']
        x = x.to(device)
        y = y.to(device)
        z_previous = None
        with torch.no_grad():
            end = time.time()
            for t in range(x.shape[1]):
                if branch_num != 0:
                    for branch_idx in range(branch_num + 1):
                        y_pred, z_previous, branch_exit = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                        if y_pred is not None:  # exit if true
                            pred_true = [y_pred.topk(1, 1, True, True)[1].detach().cpu().numpy()[0], y.item()]
                            spatial_exit[t] = branch_idx + 1
                            break
                        if branch_exit:
                            spatial_exit[t] = branch_idx + 1
                            break
                    else:
                        spatial_exit[t] = branch_num + 1
                        continue
                    if branch_exit and y_pred is None:
                        continue
                    else:
                        break
                else:
                    y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t)
                    if y_pred is not None:
                        frame_exit = t + 1
                        break
            t = time.time() - end
            print(f"the No.{itr} exit: {spatial_exit} pred_true: {pred_true}")
            print(f"the No.{itr} video costs {t}s")
            if itr != 0:
                s += t
    return s

def test_on_jetson_nano(model, device):
    s = 0
    print(f"We are on {device}")
    for i in range(110):
        t = test_random(model, device)
        print(f"Itr[{i}] time: {t:.4f}s")
        if i > 9:
            s += t
    print(f"average time: {s / 100:.4f}s")
    print(f"FPS: {1 / (s / 100 / 10)}")


def test_random(model, device):

    model = model.to(device)
    x = torch.randn(1, 10, 3, 224, 224).to(device)

    model.eval()
    branch_num = config.model.branch_num
    z_previous = None
    branch_exit = [-1 for _ in range(10)]
    frame_exit = -1
    with torch.no_grad():
        end = time.time()
        for t in range(x.shape[1]):
            if branch_num != 0:
                for branch_idx in range(branch_num + 1):
                    y_pred, z_previous, branch_exit = model(x[:, t].unsqueeze(dim=1), z_previous, t, branch_idx)
                    if y_pred is not None:  # exit if true
                        frame_exit = t
                        break
                    if branch_exit:
                        branch_exit[t] = branch_idx
                        break
                else:
                    continue
                if branch_exit and y_pred is None:
                    continue
                else:
                    break
            else:
                y_pred, z_previous = model(x[:, t].unsqueeze(dim=1), z_previous, t)
                if y_pred  is not None:
                    frame_exit = t
                    break
            print(f"frame exit: {frame_exit}")
            print(f"branch exit: {branch_exit}")
    return time.time() - end


def get_model():
    if config.model.branch_num != 0:
        model = TLEEModel(config=config, is_train=False, branch_num=config.model.branch_num)
    else:
        model = BaseModel(config=config)
    return model


def print_info():
    print(f"ckpt path: {config.test_path}")
    print(f"exit_thresh: {config.model.exit_thresh}")

if __name__ == '__main__':
    main()
    print("Done!")