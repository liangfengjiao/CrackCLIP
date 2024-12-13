import os
import json


class CrackSolver(object):
    CLSNAMES = [
        'pavement',
    ]#

    def __init__(self, root='data/crack'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                species.sort()
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}/image')
                    mask_names = os.listdir(f'{cls_dir}/{phase}/{specie}/mask')
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/image/{img_name}',
                            mask_path=f'{cls_name}/{phase}/{specie}/mask/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                    info[phase][cls_name] = cls_info


                # cls_info = []
                # #species = os.listdir(f'{cls_dir}/{phase}')
                # is_abnormal = True #True if specie not in ['good'] else False
                # img_names = os.listdir(f'{cls_dir}/{phase}/image')
                # mask_names = os.listdir(f'{cls_dir}/{phase}/mask')
                # img_names.sort()
                # mask_names.sort() if mask_names is not None else None
                # specie = 'crack'
                # for idx, img_name in enumerate(img_names):
                #     info_img = dict(
                #         img_path=f'{cls_name}/{phase}/image/{img_name}',
                #         mask_path=f'{cls_name}/{phase}/mask/{mask_names[idx]}' if is_abnormal else '',
                #         cls_name=cls_name,
                #         specie_name=specie,
                #         anomaly=1 if is_abnormal else 0,
                #     )
                #     cls_info.append(info_img)
                # info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def train_run(self, phase = "train"):
        self.meta_path = f'{self.root}/{phase}_meta.json'
        info = dict(train={}, test={})
        #info = dict()
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            cls_info = []
            # species = os.listdir(f'{cls_dir}/{phase}')
            is_abnormal = True  # True if specie not in ['good'] else False
            img_names = os.listdir(f'{cls_dir}/{phase}/image')
            mask_names = os.listdir(f'{cls_dir}/{phase}/mask')
            img_names.sort()
            mask_names.sort() if mask_names is not None else None
            specie = 'crack'
            for idx, img_name in enumerate(img_names):
                info_img = dict(
                    img_path=f'{cls_name}/{phase}/image/{img_name}',
                    mask_path=f'{cls_name}/{phase}/mask/{mask_names[idx]}' if is_abnormal else '',
                    cls_name=cls_name,
                    specie_name=specie,
                    anomaly=1 if is_abnormal else 0,
                )
                cls_info.append(info_img)
            info[phase][cls_name] = cls_info
            #info[cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

    def test_run(self, phase = "test"):
        self.meta_path = f'{self.root}/{phase}_meta.json'
        info = dict(train={}, test={})
        #info = dict()
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            cls_info = []
            # species = os.listdir(f'{cls_dir}/{phase}')
            is_abnormal = True  # True if specie not in ['good'] else False
            img_names = os.listdir(f'{cls_dir}/{phase}/image')
            mask_names = os.listdir(f'{cls_dir}/{phase}/mask')
            img_names.sort()
            mask_names.sort() if mask_names is not None else None
            specie = 'crack'
            for idx, img_name in enumerate(img_names):
                info_img = dict(
                    img_path=f'{cls_name}/{phase}/image/{img_name}',
                    mask_path=f'{cls_name}/{phase}/mask/{mask_names[idx]}' if is_abnormal else '',
                    cls_name=cls_name,
                    specie_name=specie,
                    anomaly=1 if is_abnormal else 0,
                )
                cls_info.append(info_img)
            #info[cls_name] = cls_info
            info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = CrackSolver(root='data/crack')
    runner.run()
    # runner.train_run()
    # runner.test_run()
