import argparse
import logging

import numpy as np
import pandas as pd
import torch

from src.dataset import set_infer_dataloader, set_train_dataloader
from src.model import (
    EarlyStopping,
    HepaClassifier,
    inference_one_epoch,
    train_one_epoch,
    valid_one_epoch,
)
from src.utils import data_load, seed_everything


# argparse function
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", action="store_true")

    # data setting
    parser.add_argument("--project_name", default="qxai_fold_1", type=str)
    parser.add_argument("--bio_rate", default=1.0, type=float)
    parser.add_argument("--sur_rate", default=1.0, type=float)

    # training model
    parser.add_argument("--model_arch", default="tf_efficientnet_b3", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--bs_size", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=3e-6, type=float)
    parser.add_argument("--er_patience", default=2, type=int)
    parser.add_argument("--er_mode", default="min", type=str)
    parser.add_argument("--er_monitor", default="val_loss", type=str)

    # 引数を解析
    return parser.parse_args()


def main():
    # config, global value setting
    Config = parse_arguments()
    seed_everything(Config["seed"])
    device = torch.device("cuda:0")

    # train section
    ## train dataset
    train_df = data_load(
        ver="fold",
        bio_rate=Config["bio_rate"],
        sur_rate=Config["sur_rate"],
    )
    print(f"surgery : {len(train_df[train_df.surgery == 1])}")
    print(f"biopsy : {len(train_df[train_df.surgery == 0])}")
    print()

    # logging
    if Config.log:
        logging.basicConfig(
            filename=f"./log/{Config.object}.log",
            level=logging.DEBUG,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.debug("Training Phase")
        logging.debug("practice_type : fold")
        logging.debug(f"model : {Config['model_arch']}")
        logging.debug(f"Rate of using surgery : {Config['sur_rate']}")
        logging.debug(f"Rate of using biopsy : {Config['bio_rate']}")

        logging.debug(f"surgery : {len(train_df[train_df.surgery == 1])}")
        logging.debug(f"biopsy : {len(train_df[train_df.surgery == 0])}")

    result = pd.DataFrame()

    folds = [
        [[1, 2, 3], 4, 5],
        [[2, 3, 4], 5, 1],
        [[3, 4, 5], 1, 2],
        [[4, 5, 1], 2, 3],
        [[5, 1, 2], 3, 4],
    ]

    for fold, (trn_idx, val_idx, tst_idx) in enumerate(folds):
        if Config["debug"] > 0 and fold > 0:
            break

        print(f"{fold} fold training start")

        train_loader, val_loader = set_train_dataloader(
            df=train_df,
            input_shape=Config["img_size"],
            train_bs=Config["bs_size"],
            valid_bs=Config["bs_size"],
            split=False,
            trn_idx=trn_idx,
            val_idx=val_idx,
        )

        ## train model
        model = HepaClassifier(
            model_arch=Config["model_arch"],
            pretrained=Config["pretrained"],
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Config["lr"], weight_decay=Config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=Config["min_lr"],
            last_epoch=-1,
        )
        er = EarlyStopping(Config["er_patience"])
        loss_tr = torch.nn.CrossEntropyLoss().to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        ## training start
        for epoch in range(Config["epochs"]):
            train_one_epoch(
                epoch,
                model,
                loss_tr,
                optimizer,
                train_loader,
                device,
                scheduler=scheduler,
                schd_batch_update=False,
            )

            with torch.no_grad():
                monitor = valid_one_epoch(
                    epoch,
                    model,
                    loss_fn,
                    val_loader,
                    device,
                    scheduler=None,
                    schd_loss_update=False,
                )

            # Early Stopping
            if er.update(monitor[Config["er_monitor"]], epoch, Config["er_mode"]) < 0:
                break
            if epoch == er.val_epoch:
                torch.save(
                    model.state_dict(),
                    f"./save/{Config['folder_name']}_{fold + 1}_{epoch}",
                )

        del model, optimizer, train_loader, val_loader, scheduler
        torch.cuda.empty_cache()

        if Config.log:
            logging.debug(f"Best Epoch({fold + 1}) : {er.val_epoch}")

        # infer section
        ## infer dataset
        test_df = data_load(
            ver="fold",
            bio_rate=Config["bio_rate"],
            sur_rate=Config["sur_rate"],
        )

        tst_loader, tst_df, val_loader, val_df = set_infer_dataloader(
            df=test_df,
            input_shape=Config["img_size"],
            valid_bs=Config["bs_size"],
            split=False,
            val_idx=val_idx,
            tst_idx=tst_idx,
        )

        ## infer model
        model = HepaClassifier(
            model_arch=Config["model_arch"],
            pretrained=Config["pretrained"],
        ).to(device)
        model.load_state_dict(
            torch.load(f"./save/{Config['folder_name']}_{fold + 1}_{er.val_epoch}")
        )

        tst_preds = []
        val_preds = []

        with torch.no_grad():
            tst_preds += [inference_one_epoch(model, tst_loader, device)]
            val_preds += [inference_one_epoch(model, val_loader, device)]

        tst_preds = np.mean(tst_preds, axis=0)
        val_preds = np.mean(val_preds, axis=0)

        del model, er
        torch.cuda.empty_cache()

        tst_df = tst_df[["img_path", "label"]]
        tst_df["result"] = tst_preds[:, 0]
        tst_df["fold"] = fold + 1
        tst_df["ver"] = "test"

        val_df = val_df[["img_path", "label"]]
        val_df["result"] = val_preds[:, 0]
        val_df["fold"] = fold + 1
        val_df["ver"] = "val"

        result = pd.concat([result, tst_df])
        result = pd.concat([result, val_df])

    # 予測結果を保存
    folder_name = Config["folder_name"]
    folder_path = "./result/preds"

    result.to_csv(f"{folder_path}/{folder_name}_preds.csv", index=False)
    if Config.log:
        logging.debug(f"Result filename : {folder_name}_preds.csv")


if __name__ == "__main__":
    main()
