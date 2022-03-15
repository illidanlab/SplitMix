"""Split-Mix Federated Learning"""
import sys, os, argparse, copy, time
import numpy as np
import wandb
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn.modules.batchnorm import _NormBase
# federated
from federated.learning import train_slimmable, test, fed_test_model, refresh_bn, test_dbn
# model and data
from nets.models import ScalableModule
from nets.slimmable_models import EnsembleNet, EnsembleSubnet
# utils
from utils.utils import set_seed, AverageMeter, CosineAnnealingLR, \
    MultiStepLR, LocalMaskCrossEntropyLoss, str2bool
from utils.config import CHECKPOINT_ROOT

# NOTE import desired federation
from federated.core import SplitFederation as Federation, AdversaryCreator


def render_run_name(args, exp_folder):
    """Return a unique run_name from given args."""
    if args.model == 'default':
        args.model = {'Digits': 'ens_digit', 'Cifar10': 'ens_preresnet18', 'DomainNet': 'ens_alex'}[args.data]
    run_name = f'{args.model}'
    run_name += Federation.render_run_name(args)
    # log non-default args
    if args.seed != 1: run_name += f'__seed_{args.seed}'
    # opt
    if args.lr_sch != 'none': run_name += f'__lrs_{args.lr_sch}'
    if args.opt != 'sgd': run_name += f'__opt_{args.opt}'
    if args.batch != 32: run_name += f'__batch_{args.batch}'
    if args.wk_iters != 1: run_name += f'__wk_iters_{args.wk_iters}'
    # slimmable
    if args.no_track_stat: run_name += f"__nts"
    # split-mix
    if not args.rescale_init: run_name += '__nri'
    if not args.rescale_layer: run_name += '__nrl'
    if args.loss_temp != 'none': run_name += f'__lt{args.loss_temp}'
    if args.lbn: run_name += '__lbn'
    # adv train
    if args.adv_lmbd > 0:
        run_name += f'__at{args.adv_lmbd}'

    args.save_path = os.path.join(CHECKPOINT_ROOT, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_FILE = os.path.join(args.save_path, run_name)
    return run_name, SAVE_FILE


def get_model_fh(data, model, atom_slim_ratio):
    # FIXME Only use EnsembleNet or Slimmable model.
    if data == 'Digits':
        if model in ['digit']:
            from nets.slimmable_models import SlimmableDigitModel
            # TODO remove. Function the same as ens_digit
            ModelClass = SlimmableDigitModel
        elif model == 'ens_digit':
            from nets.models import DigitModel
            ModelClass = lambda **kwargs: EnsembleNet(
                base_net=DigitModel, atom_slim_ratio=atom_slim_ratio,
                rescale_init=args.rescale_init, rescale_layer=args.rescale_layer, **kwargs)
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data in ['DomainNet']:
        if model in ['alex']:
            from nets.slimmable_models import SlimmableAlexNet
            ModelClass = SlimmableAlexNet
        elif model == 'ens_alex':
            from nets.models import AlexNet
            ModelClass = lambda **kwargs: EnsembleNet(
                base_net=AlexNet, atom_slim_ratio=atom_slim_ratio,
                rescale_init=args.rescale_init, rescale_layer=args.rescale_layer, **kwargs)
        else:
            raise ValueError(f"Invalid model: {model}")
    elif data == 'Cifar10':
        if model in ['preresnet18']:  # From heteroFL
            from nets.HeteFL.slimmable_preresne import resnet18
            ModelClass = resnet18
        elif model in ['ens_preresnet18']:
            if args.no_track_stat:
                # FIXME remove on release
                from nets.HeteFL.preresne import resnet18
            else:
                from nets.HeteFL.preresnet import resnet18
            ModelClass = lambda **kwargs: EnsembleNet(
                base_net=resnet18, atom_slim_ratio=atom_slim_ratio,
                rescale_init=args.rescale_init, rescale_layer=args.rescale_layer, **kwargs)
        else:
            raise ValueError(f"Invalid model: {model}")
    else:
        raise ValueError(f"Unknown dataset: {data}")
    return ModelClass


def fed_test(fed, running_model, verbose, adversary=None, val_mix_model=None):
    mark = 's' if adversary is None else 'r'
    val_acc_list = [None for _ in range(fed.client_num)]
    val_loss_mt = AverageMeter()
    slim_val_acc_mt = {slim_ratio: AverageMeter() for slim_ratio in fed.val_slim_ratios}
    for client_idx in range(fed.client_num):
        fed.download(running_model, client_idx)

        for i_slim_ratio, slim_ratio in enumerate(fed.val_slim_ratios):
            # Load and set slim ratio
            if isinstance(running_model, EnsembleNet):
                running_model.switch_slim_mode(slim_ratio)
                val_mix_model = running_model
            else:
                # FIXME ad-hoc for SlimmableNet
                running_model.switch_slim_mode(1.0)  # full net should load the full net
                val_mix_model.full_net.load_state_dict(running_model.state_dict())
                val_mix_model.set_total_slim_ratio(slim_ratio)

            # Test
            if running_model.bn_type.startswith('d'):
                val_loss, val_acc = test_dbn(val_mix_model, val_loaders[client_idx], loss_fun, device,
                                             adversary=adversary, att_BNn=True, detector='gt')
            else:
                val_loss, val_acc = test(val_mix_model, val_loaders[client_idx], loss_fun, device,
                                         adversary=adversary)

            # Log
            val_loss_mt.append(val_loss)
            val_acc_list[client_idx] = val_acc  # NOTE only record the last slim_ratio.
            if verbose > 0:
                print(' {:<19s} slim {:.2f}| Val {:s}Loss: {:.4f} | Val {:s}Acc: {:.4f}'.format(
                    'User-' + fed.clients[client_idx] if i_slim_ratio == 0 else ' ', slim_ratio,
                    mark.upper(), val_loss, mark.upper(), val_acc))
            wandb.log({
                f"{fed.clients[client_idx]} sm{slim_ratio:.2f} val_s-acc": val_acc,
            }, commit=False)
            if slim_ratio == fed.user_max_slim_ratios[client_idx]:
                wandb.log({
                    f"{fed.clients[client_idx]} val_{mark}-acc": val_acc,
                }, commit=False)
            slim_val_acc_mt[slim_ratio].append(val_acc)
    slim_val_acc_dict = {k: mt.avg if len(mt) > 0 else None for k, mt in slim_val_acc_mt.items()}
    wandb.log({
        f"slim{k:.2f} val_sacc": acc for k, acc in slim_val_acc_dict.items()
    }, commit=False)
    return val_acc_list, val_loss_mt.avg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.seterr(all='raise')  # make sure warning are raised as exception

    parser = argparse.ArgumentParser()
    # basic problem setting
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data', type=str, default='Digits', help='data name')
    parser.add_argument('--model', type=str.lower, default='default', help='model name')
    parser.add_argument('--no_track_stat', action='store_true', help='disable BN tracking')
    parser.add_argument('--test_refresh_bn', action='store_true', help='refresh BN before test')
    # control
    parser.add_argument('--no_log', action='store_true', help='disable wandb log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--verbose', type=int, default=0, help='verbose level: 0 or 1')
    # federated
    Federation.add_argument(parser)
    # optimization
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_sch', type=str, default='none', help='learning rate schedule')
    parser.add_argument('--opt', type=str.lower, default='sgd', help='optimizer')
    parser.add_argument('--iters', type=int, default=300, help='#iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1, help='#epochs in local train')
    # slimmable test
    parser.add_argument('--test_slim_ratio', type=float, default=1.,
                        help='slim_ratio of model at testing.')
    parser.add_argument('--sort_bases', action='store_true', help='sort base models by val acc.')
    # split-mix
    parser.add_argument('--rescale_init', type=str2bool, default=True, help='rescale init after slim')
    parser.add_argument('--rescale_layer', type=str2bool, default=True, help='rescale layer outputs after slim')
    parser.add_argument('--loss_temp', type=str, default='none',
                        help='temper cross-entropy loss (str|float):'
                             ' auto - set temp as the width scale; none - no temper; '
                             'other float values.')
    parser.add_argument('--lbn', type=str2bool, default=False, help='use client-local BN stats (valid if tracking stats)')
    # adversarial train
    parser.add_argument('--adv_lmbd', type=float, default=0.,
                        help='adv coefficient in [0,1]; default 0 for standard training.')
    parser.add_argument('--test_noise', choices=['none', 'LinfPGD'], default='none')
    parser.add_argument('--test_adv_lmbd', type=float, default=0.)
    args = parser.parse_args()

    set_seed(args.seed)

    # set experiment files, wandb
    exp_folder = f'SplitMix_{args.data}'
    run_name, SAVE_FILE = render_run_name(args, exp_folder)
    wandb.init(group=run_name[:120], project=exp_folder, mode='offline' if args.no_log else 'online',
               config={**vars(args), 'save_file': SAVE_FILE})


    # /////////////////////////////////
    # ///// Fed Dataset and Model /////
    # /////////////////////////////////
    fed = Federation(args.data, args)
    # Data
    train_loaders, val_loaders, test_loaders = fed.get_data()
    mean_batch_iters = int(np.mean([len(tl) for tl in train_loaders]))
    print(f"  mean_batch_iters: {mean_batch_iters}")

    # Model
    ModelClass = get_model_fh(args.data, args.model, args.atom_slim_ratio)
    running_model = ModelClass(
        track_running_stats=not args.no_track_stat or (args.test and args.test_refresh_bn), num_classes=fed.num_classes,
        bn_type='dbn' if 0. < args.adv_lmbd < 1. else 'bn',
        slimmable_ratios=fed.train_slim_ratios,
    ).to(device)
    # mixed model for validation.
    val_mix_model = running_model if isinstance(running_model, EnsembleNet) \
        else EnsembleSubnet(copy.deepcopy(running_model), args.atom_slim_ratio)

    # adversary
    if args.adv_lmbd > 0. or args.test:
        assert isinstance(running_model, EnsembleNet), "Did not create adv for val_mix_model"
        make_adv = AdversaryCreator(args.test_noise if args.test else 'LinfPGD')
        adversary = make_adv(running_model)
    else:
        adversary = None

    # Loss
    if args.pu_nclass > 0:  # niid
        loss_fun = LocalMaskCrossEntropyLoss(fed.num_classes)
    else:
        loss_fun = nn.CrossEntropyLoss()

    # Use running model to init a fed aggregator
    fed.make_aggregator(running_model, local_bn=args.lbn)


    # /////////////////
    # //// Resume /////
    # /////////////////
    # log the best for each model on all datasets
    best_epoch = 0
    best_acc = [0. for j in range(fed.client_num)]
    train_elapsed = [[] for _ in range(fed.client_num)]
    start_epoch = 0
    if args.resume or args.test:
        if os.path.exists(SAVE_FILE):
            print(f'Loading chkpt from {SAVE_FILE}')
            checkpoint = torch.load(SAVE_FILE)
            best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
            train_elapsed = checkpoint['train_elapsed']
            start_epoch = int(checkpoint['a_iter']) + 1
            fed.model_accum.load_state_dict(checkpoint['server_model'])

            print('Resume training from epoch {} with best acc:'.format(start_epoch))
            for client_idx, acc in enumerate(best_acc):
                print(' Best user-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                    fed.clients[client_idx], best_epoch, acc))
        else:
            if args.test:
                raise FileNotFoundError(f"Not found checkpoint at {SAVE_FILE}")
            else:
                print(f"Not found checkpoint at {SAVE_FILE}\n **Continue without resume.**")


    # ///////////////
    # //// Test /////
    # ///////////////
    if args.test:
        wandb.summary[f'best_epoch'] = best_epoch
        # wandb.summary[f'per_epoch_train_elapsed'] = np.sum([np.mean(client_ts) for client_ts in train_elapsed])

        # val to select base models
        if args.sort_bases and isinstance(running_model, EnsembleNet):
            base_accs = []
            print(f"Evaluate base models..")
            for base_idx in tqdm(range(fed.num_base), file=sys.stdout):
                running_model.switch_slim_mode(fed.args.atom_slim_ratio, base_idx)
                val_acc = fed_test_model(fed, running_model, val_loaders, loss_fun, device)
                base_accs.append(val_acc)
            print(f" Base Accs: {', '.join([f'{a:.3f}' for a in base_accs])}")
            base_idxs = np.argsort(base_accs)[::-1]
            print(f" Sorted base indexes: {base_idxs}")
            running_model.base_idxs = base_idxs
            # fed.download()

        # Set up model with specified width
        print(f"  Test model: {args.model} x{args.test_slim_ratio} lmbd{args.test_adv_lmbd}"
              + ('' if args.test_noise == 'none' else f' with {args.test_noise} noise'))
        assert args.atom_slim_ratio > 0, "When ensemble, the atom ratio has to be defined by" \
                                         f" args.slim_ratio > 0. But got {args.atom_slim_ratio}"
        print(f"  Ensemble {int(args.test_slim_ratio / args.atom_slim_ratio)} "
              f"{args.atom_slim_ratio} base nets")
        if not isinstance(running_model, EnsembleNet):
            assert args.adv_lmbd == 0, "Not create adversary for EnsembleSubnet."
            running_model.switch_slim_mode(1.)
            test_model = EnsembleSubnet(running_model, subnet_ratio=args.atom_slim_ratio,
                                        ensemble_num=int(
                                            args.test_slim_ratio / args.atom_slim_ratio))
        else:
            running_model.switch_slim_mode(args.test_slim_ratio)
            test_model = running_model

        # Test on clients
        if isinstance(running_model, EnsembleNet):
            print(f"### current slice: {running_model.current_slice()}")
        test_acc_mt = AverageMeter()
        for test_idx, test_loader in enumerate(test_loaders):
            fed.download(running_model, test_idx, strict=not args.test_refresh_bn)
            if running_model.bn_type.startswith('d'):
                _, test_acc = test_dbn(test_model, test_loader, loss_fun, device,
                                       adversary=adversary,
                                       detector='clean',  # FIXME does this really matter?
                                       att_BNn=True,  # args.te_att_BNn,  # FIXME we shall remove this since we will attack the mixed output.
                                       adversary_name=args.test_noise,
                                       mix_dual_logit_lmbd=args.test_adv_lmbd,
                                       attack_mix_dual_logit_lmbd=args.test_adv_lmbd,
                                       deep_mix=True,
                                       )
            else:
                if args.test_refresh_bn:
                    # test_model.base_net.rescale_layer = False
                    def set_rescale_layer_and_bn(m):
                        if isinstance(m, ScalableModule):
                            m.rescale_layer = False
                        if isinstance(m, _NormBase):
                            m.reset_running_stats()
                            m.momentum = None
                    test_model.apply(set_rescale_layer_and_bn)
                    for ep in tqdm(range(20), desc='refresh bn', leave=False):
                        refresh_bn(test_model, train_loaders[test_idx], device)
                _, test_acc = test(test_model, test_loader, loss_fun, device, adversary=adversary)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(fed.clients[test_idx], test_acc))

            wandb.summary[f'{fed.clients[test_idx]} test acc'] = test_acc
            test_acc_mt.append(test_acc)

        # Profile model FLOPs, sizes (#param)
        from nets.profile_func import profile_model
        flops, params = profile_model(test_model, device=device)
        wandb.summary['GFLOPs'] = flops / 1e9
        wandb.summary['model size (MB)'] = params / 1e6
        print('GFLOPS: %.4f, model size: %.4fMB' % (flops / 1e9, params / 1e6))

        print(f"\n Average Test Acc: {test_acc_mt.avg}")
        wandb.summary[f'avg test acc'] = test_acc_mt.avg
        wandb.finish()

        exit(0)


    # ////////////////
    # //// Train /////
    # ////////////////
    # LR scheduler
    if args.lr_sch == 'cos':
        lr_sch = CosineAnnealingLR(args.iters, eta_max=args.lr, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step':
        lr_sch = MultiStepLR(args.lr, milestones=[150, 250], gamma=0.1, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step50':
        lr_sch = MultiStepLR(args.lr, milestones=[150+50, 250+50], gamma=0.1, last_epoch=start_epoch)
    elif args.lr_sch == 'multi_step100':
        lr_sch = MultiStepLR(args.lr, milestones=[150+100, 250+100], gamma=0.1, last_epoch=start_epoch)
    else:
        assert args.lr_sch == 'none', f'Invalid lr_sch: {args.lr_sch}'
        lr_sch = None
    shift_tr_cnt_mt = [0 for _ in range(fed.num_base)]  # count of trained times for each base model
    for a_iter in range(start_epoch, args.iters):
        # set global lr
        global_lr = args.lr if lr_sch is None else lr_sch.step()
        wandb.log({'global lr': global_lr}, commit=False)

        # ----------- Train Client ---------------
        train_loss_mt, train_acc_mt = AverageMeter(), AverageMeter()
        print("============ Train epoch {} ============".format(a_iter))
        for client_idx in fed.client_sampler.iter():
            # (Alg 2) Sample base models defined by shift index.
            slim_ratios, slim_shifts = fed.sample_bases(client_idx)

            start_time = time.process_time()

            # Download global model to local
            fed.download(running_model, client_idx)

            # (Alg 3) Local Train
            if args.opt == 'sgd':
                optimizer = optim.SGD(params=running_model.parameters(), lr=global_lr,
                                      momentum=0.9, weight_decay=5e-4)
            elif args.opt == 'adam':
                optimizer = optim.Adam(params=running_model.parameters(), lr=global_lr)
            else:
                raise ValueError(f"Invalid optimizer: {args.opt}")
            local_iters = mean_batch_iters * args.wk_iters if args.partition_mode != 'uni' \
                else len(train_loaders[client_idx]) * args.wk_iters
            train_loss, train_acc = train_slimmable(
                running_model, train_loaders[client_idx], optimizer, loss_fun, device,
                max_iter=local_iters,
                slim_ratios=slim_ratios, slim_shifts=slim_shifts, progress=args.verbose > 0,
                loss_temp=args.loss_temp,
                adversary=adversary, adv_lmbd=args.adv_lmbd, att_BNn=True,
            )

            # Upload
            fed.upload(running_model, client_idx,
                       max_slim_ratio=max(slim_ratios), slim_bias_idx=slim_shifts)

            # Log
            client_name = fed.clients[client_idx]
            elapsed = time.process_time() - start_time
            wandb.log({f'{client_name}_train_elapsed': elapsed}, commit=False)
            train_elapsed[client_idx].append(elapsed)

            train_loss_mt.append(train_loss), train_acc_mt.append(train_acc)
            for slim_shift in slim_shifts:
                shift_tr_cnt_mt[slim_shift] += 1
            print(f' User-{client_name:<10s} Train | Loss: {train_loss:.4f} |'
                  f' Acc: {train_acc:.4f} | Elapsed: {elapsed:.2f} s')
            wandb.log({
                f"{client_name} train_loss": train_loss,
                f"{client_name} train_acc": train_acc,
            }, commit=False)

        # Use accumulated model to update server model
        fed.aggregate()


        # ----------- Validation ---------------
        val_acc_list, val_loss = fed_test(
            fed, running_model, args.verbose, val_mix_model=val_mix_model, adversary=None)
        if args.adv_lmbd > 0:
            print(f' Avg Val SAcc {np.mean(val_acc_list) * 100:.2f}%')
            wandb.log({'val_sacc': np.mean(val_acc_list)}, commit=False)
            val_racc_list, val_rloss = fed_test(
                fed, running_model, args.verbose, val_mix_model=val_mix_model, adversary=adversary)
            print(f' Avg Val RAcc {np.mean(val_racc_list) * 100:.2f}%')
            wandb.log({'val_racc': np.mean(val_racc_list)}, commit=False)

            val_acc_list = [(1-args.adv_lmbd) * sa_ + args.adv_lmbd * ra_
                            for sa_, ra_ in zip(val_acc_list, val_racc_list)]
            val_loss = (1-args.adv_lmbd) * val_loss + args.adv_lmbd * val_rloss
        # Log averaged
        print(f' [Overall] Train Loss {train_loss_mt.avg:.4f} Acc {train_acc_mt.avg*100:.1f}% '
              f'| Val Acc {np.mean(val_acc_list)*100:.2f}%')
        wandb.log({
            f"train_loss": train_loss_mt.avg,
            f"train_acc": train_acc_mt.avg,
            f"val_loss": val_loss,
            f"val_acc": np.mean(val_acc_list),
        }, commit=False)
        wandb.log({
            f"shift{s} train cnt": cnt for s, cnt in enumerate(shift_tr_cnt_mt)
        }, commit=False)


        # ----------- Save checkpoint -----------
        if np.mean(val_acc_list) > np.mean(best_acc):
            best_epoch = a_iter
            for client_idx in range(fed.client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                if args.verbose > 0:
                    print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(
                          fed.clients[client_idx], best_epoch, best_acc[client_idx]))
            print(' [Best Val] Acc {:.4f}'.format(np.mean(val_acc_list)))

            # Save
            print(f' Saving the local and server checkpoint to {SAVE_FILE}')
            save_dict = {
                'server_model': fed.model_accum.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter,
                'all_domains': fed.all_domains,
                'train_elapsed': train_elapsed,
            }
            torch.save(save_dict, SAVE_FILE)
        wandb.log({
            f"best_val_acc": np.mean(best_acc),
        }, commit=True)
