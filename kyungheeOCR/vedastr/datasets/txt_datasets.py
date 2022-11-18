import os
import io
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class TxtDataset(BaseDataset):

    def __init__(self, root, gt_txt, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=25):
        super(TxtDataset, self).__init__(root, gt_txt, transform, character=character,
                                         batch_max_length=batch_max_length)

    def get_name_list(self):
        with io.open(self.gt_txt, 'r', encoding="utf-8-sig") as gt:
            for line in gt.readlines():
                try:
                    img_name, label = line.strip().split('\t')
                except ValueError:
                    continue
                if self.filter(label):
                    continue
                else:
                    self.img_names.append(os.path.join(self.root, img_name))
                    self.gt_texts.append(label)

        self.samples = len(self.gt_texts)
