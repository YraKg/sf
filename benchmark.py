import torch
from  evaluator import *
from utils import get_data_loader
from model import create_model
import time
from autoattack import AutoAttack

checkpoint_path = None#'./snapshots/wrn_k256_e32_ni100_adv/checkpoint_5.pth'#'./snapshots/FT_swin-L_adv_ni50/checkpoint_3.pth'#"./snapshots/FT_wrn34_20_k256_e32_ni100_adv/checkpoint_1.pth"#"./snapshots/FT_wrn34_20_k256_e32_ni100_adv/checkpoint_5.pth"#"./snapshots/wrn_k256_e32_ni100_adv/checkpoint_25.pth"#"./snapshots/dynamic2_train_batched_ni20_adv/checkpoint_100.pth"
#"./snapshots/preact_k256_e32_ni100_adv/checkpoint_50.pth" has very good e=1,2,4


def get_model(name, checkpoint_path=None, num_classes = 10):
    model = create_model(name=name, num_classes=num_classes)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
    return model.cuda()


class AttackWrapper():
    def __init__(self,model,attack):
        self.model = model
        self.attack = attack
class AutoAttackWrapper(AttackWrapper):
    def __init__(self,model,attack):
        super().__init__(model,attack) 
        
    def perturb(self,x,y):
        x_adv = self.attack.run_standard_evaluation(x, y, bs=y.shape[0])
        return x_adv,None,None
def get_attack(model,type = "inf",epsilon = 8):
    if type == "inf":
        epsilon = epsilon/255
        print("attacking with epsilon - ",epsilon)
        adversary = AutoAttack(model, norm='Linf',verbose = False, eps=epsilon, version='custom', attacks_to_run = ['apgd-ce'],log_path = './log.txt')
        return AutoAttackWrapper(model,adversary)
    return None
def eval_L_inf(model_name = "preactresnet",epsilons=[4],dataset='cifar10'):
    print(f"evaluating experiment: {checkpoint_path}")
    
    train_loader, test_loader, num_classes, img_size, train_set, test_set = get_data_loader(
        None,
        data_name=dataset,
        data_dir=f"./data/{dataset}",
        batch_size = 16,
        test_batch_size=16,
        eval_samples=1000,
        num_workers=4)
    print(len(test_set))
    model = get_model(checkpoint_path=checkpoint_path, name=model_name)
    #model = create_model(name="wideresnet70_16", num_classes=num_classes).cuda()
    


    counter = 0
    for batch in test_loader:
        x,y = batch
        out = model(x.cuda())
    
    
        y_t = torch.argmax(out,dim=-1)
        if (y_t > 100).any():
            print("yahsa lava")
        counter += (y_t == y.cuda()).sum()

    print("clean_acc=",counter/len(test_set))
    # t = (torch.rand(len(test_set)) < 1/20)
    # from itertools import compress
    # test_set = list(compress(test_set,t.tolist()))
    # print(len(test_set))
    # test_loader = torch.utils.data.DataLoader(test_set,batch_size=32,shuffle=True)
    

    for eps in epsilons:
        attack = get_attack(model,type = "inf",epsilon = eps)
        ev = Evaluator(model,attack,verbose=True) 
        clean_acc, robust_acc = ev.evaluate(test_loader)
        
        print(f"Epsilon {eps}: Robust Acc {round(robust_acc*100,2)}%")

    print(f"Clean accuracy: {round(clean_acc*100,2)}%")

if __name__ == "__main__":
    eval_L_inf(model_name="FT_swin-L",dataset='imagenet100')