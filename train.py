import argparse, pickle, time, sys
import torch
import torch.nn as nn
from data import train_gen

torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network trainer.")
    parser.add_argument("-v", "--version", action="version", version="train 1.0")
    parser.add_argument("-n", "--network", help="path to saved network")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-f", "--frequency", type=int, default=1000,
                        help="number of batches until printing a summary")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-b", "--batches", type=int, default=10,
                        help="number of batches to evalulate average loss")
    parser.add_argument("command", choices=["train", "test"])
    args = parser.parse_args()

    loss_fn = nn.MSELoss()

    # evaulate model's average performance
    if args.command == "test":
        G = torch.load(args.network).cuda()
        G.requires_grad_(False)
        loss = 0
        for b, (batchX, batchy) in enumerate(train_gen):
            if b >= args.batches:
                break
            batchX, batchy = batchX.cuda(), batchy.cuda()
            y_pred = G(2*batchX - 1, None)
            y_pred = 127.5*y_pred.permute(0, 2, 3, 1) + 128
            loss += loss_fn(y_pred, batchy.to(torch.float)).item()
        print(f"average loss: {loss/args.batches:.3f} ")
        exit()

    # train model
    with open(args.network, "rb") as f:
        data = pickle.load(f)
        G, D = data["G_ema"].cuda(), data["D"].cuda()

    G.training = G.mapping.training = G.synthesis.training = True
    optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate)

    batches, total_loss, start = 0, 0, time.time()
    for epoch in range(args.epochs):
        for batch, (batchX, batchy) in enumerate(train_gen):
            batchX, batchy = batchX.cuda(), batchy.cuda()
            optimizer.zero_grad()
            G.requires_grad_(True)
            # forward
            y_pred = G(2*batchX - 1, None)
            y_pred = 127.5*y_pred.permute(0, 2, 3, 1) + 128
            loss = loss_fn(y_pred, batchy.to(torch.float))

            total_loss += loss.item()
            batches += 1

            # backward
            G.requires_grad_(False)
            loss.backward()
            optimizer.step()

            # summary
            if batches == args.frequency:
                torch.save(G, f"train/model{epoch}_{batch}.net")

                print(f"epoch: {epoch}, batch: {batch}, loss: {total_loss/args.frequency:.3f}")
                print(f"took {time.time() - start} seconds")
                sys.stdout.flush()
                batches, total_loss, start = 0, 0, time.time()

