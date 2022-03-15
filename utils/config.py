import os
def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)
hostname = os.uname()[1]  # type: str
# Update your paths here.
CHECKPOINT_ROOT = './checkpoint'
if int(hostname.split('-')[-1]) >= 8:
    data_root = '/localscratch2/jyhong/'
elif hostname.startswith('illidan'):
    data_root = '/media/Research/jyhong/data'
else:
    data_root = './data'
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)

if hostname.startswith('illidan') and int(hostname.split('-')[-1]) < 8:
    # personal config
    home_path = os.path.expanduser('~/')
    DATA_PATHS = {
        "Digits": home_path + "projects/FedBN/data",
        "DomainNet": data_root + "/DomainNet",
        "DomainNetPathList": home_path + "projects/FedBN/data/",  # store the path list file from FedBN
        "Cifar10": data_root,
    }
else:
    DATA_PATHS = {
        "Digits": data_root + "/Digits",
        "DomainNet": data_root + "/DomainNet",
        "DomainNetPathList": data_root + "/DomainNet/domainnet10/",  # store the path list file from FedBN
        "Cifar10": data_root,
    }
