import numpy as np

class AveragePrecision:
    def __init__(self, num_classes, video_num, frame_num, branch_num) -> None:
        self.num_classes = num_classes
        self.actual = []
        self.pred = []
        self.exit_info = np.zero((video_num, frame_num))

    def update(self, x):
        act, pred, exit_info_per_video, idx = x
        self.actual.append(act)
        self.pred.append(pred)
        self.exit_info[idx] = exit_info_per_video
    
    def compute_precision_recall(self,):
        classes_ap = [{'TP': 0, 'FP': 0, 'FN': 0, 'precision': 0, 'recall': 0, 'AP': 0} for  _ in range(self.num_classes)]
        for act, pred in zip(self.actual, self.pred):
            if act == pred:
                classes_ap[act]['TP'] += 1
            else:
                classes_ap[act]['FN'] += 1
                classes_ap[pred]['FP'] += 1
        for cls in range(self.num_classes):
            precision, recall = 0, 0
            TP, FP, FN = classes_ap['TP'], classes_ap['FP'], classes_ap['FN']
            precision = TP / (FP + TP + 1e-15)
            recall = TP / (TP + FN + 1e-15)
            classes_ap[cls]['precision'] = precision
            classes_ap[cls]['recall'] = recall
        return classes_ap
    
