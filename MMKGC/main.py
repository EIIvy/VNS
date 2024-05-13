import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from IPython import embed
import wandb
from utils import setup_parser
from utils.tools import *
from data.Sampler import *
import pickle
def main():
    parser = setup_parser() #设置参数
    args = parser.parse_args()
    if args.load_config:
        args = load_config(args, args.config_path)
    seed_everything(args.seed) 
    """set up sampler to datapreprocess""" #设置数据处理的采样过程
    train_sampler_class = import_class(f"data.{args.train_sampler_class}")
    train_sampler = train_sampler_class(args)  # 这个sampler是可选择的
    #print(train_sampler)
    test_sampler_class = import_class(f"data.{args.test_sampler_class}")
    test_sampler = test_sampler_class(train_sampler)  # test_sampler是一定要的
    """set up datamodule""" #设置数据模块
    data_class = import_class(f"data.{args.data_class}") #定义数据类 DataClass
    kgdata = data_class(args, train_sampler, test_sampler)
    """set up model"""
    model_class = import_class(f"model.{args.model_name}")
    if args.dataset_name =='YG15K':
        file_path = "dataset/" +args.dataset_name + "/YG15K_id_img_feature_dict_"+str(args.IMG)+".pkl"
    elif args.dataset_name =='FB15K':
        file_path = "dataset/" +args.dataset_name + "/FB15K_id_img_feature_"+str(args.IMG)+".pkl"
    elif args.dataset_name == 'DB15K':
        file_path = "dataset/" +args.dataset_name + "/DB15K_id_img_feature_"+str(args.IMG)+".pkl"

    
    with open(file_path, 'rb') as file:
        img_feature_dict = pickle.load(file)
    # 获取所有图像嵌入的列表
    embedding_list = list(img_feature_dict.values())
    # 转换列表为 PyTorch 张量
    img_emb = torch.tensor(embedding_list).float()
    if args.dataset_name == 'YG15K':
        rel = pickle.load(open("dataset/YG15K/YG15K_rel_img_feature_dict_3.pkl","rb"))
    elif args.dataset_name == 'FB15K':
        rel = pickle.load(open("dataset/FB15K/FB15K_rel_img_feature_dict_1.pkl","rb"))
    elif args.dataset_name == 'DB15K':
        rel = pickle.load(open("dataset/DB15K/DB15K_rel_img_feature_dict"+str(args.rel_number)+".pkl","rb"))
    rel1 = np.array(list(rel.values()))
    if sum(sum(np.isnan(rel1))) > 0:
        rel1 = np.nan_to_num(rel1, nan=0, posinf=0, neginf=0)

    rel_emb = torch.tensor(rel1).float()
    model = model_class(args, img_emb, rel_emb)
    
    """set up lit_model"""
    litmodel_class = import_class(f"lit_model.{args.litmodel_name}")
    logger1 = get_logger(args)
    lit_model = litmodel_class(model, args, logger1)
    """set up logger"""
    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.use_wandb:
        log_name = "_".join([args.model_name, args.dataset_name, str(args.lr)])
        logger = pl.loggers.WandbLogger(name=log_name, project="NeuralKG")
        logger.log_hyperparams(vars(args))
    """early stopping"""
    early_callback = pl.callbacks.EarlyStopping(
        monitor="Eval|mrr",
        mode="max",
        patience=args.early_stop_patience,
        # verbose=True,
        check_on_train_epoch_end=False,
    )
    """set up model save method"""
    # 目前是保存在验证集上mrr结果最好的模型
    # 模型保存的路径
    dirpath = "/".join(["output", args.eval_task, args.dataset_name, args.model_name])
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="Eval|mrr",
        mode="max",
        filename="{epoch}-{Eval|mrr:.3f}",
        dirpath=dirpath,
        save_weights_only=True,
        save_top_k=1,
    )
    callbacks = [early_callback, model_checkpoint]
    # initialize trainer
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        default_root_dir="training/logs",
        gpus="0,",
        check_val_every_n_epoch=args.check_per_epoch,
    )
    '''保存参数到config'''
    if args.save_config:
        save_config(args)
    if args.use_wandb:
        logger.watch(lit_model)
    if not args.test_only:
        # train&valid
        trainer.fit(lit_model, datamodule=kgdata)
        # 加载本次实验中dev上表现最好的模型，进行test
        path = model_checkpoint.best_model_path
    else:
        path = args.checkpoint_dir
    lit_model.load_state_dict(torch.load(path)["state_dict"])
    lit_model.eval()
    trainer.test(lit_model, datamodule=kgdata)

if __name__ == "__main__":
    main()
