import argparse, pickle, time
import torch
import torch.nn as nn
from data import train_gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network trainer.")
    parser.add_argument("-v", "--version", action="version", version="train 1.0")
    parser.add_argument("-n", "--network", help="path to saved network")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-f", "--frequency", type=int, default=10,
                        help="number of batches until printing a summary")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    with open(args.network, "rb") as f:
        data = pickle.load(f)
        G, D = data["G_ema"].cuda(), data["D"].cuda()

    G.requires_grad_(True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate)

    batches, total_loss, start = 0, 0, time.time()
    for epoch in range(args.epochs):
        for batch, (batchX, batchy) in enumerate(train_gen):
            optimizer.zero_grad()
            batchX, batchy = batchX.cuda(), batchy.cuda()
            # forward
            y_pred = G(batchX, None)
            y_pred = 127.5*y_pred.permute(0, 2, 3, 1) + 128
            loss = loss_fn(y_pred, batchy.to(torch.float))

            total_loss += loss.item()
            batches += 1

            # backward
            loss.backward()
            optimizer.step()

            # summary
            if batches == args.frequency:
                torch.save(G, f"train/model{epoch}_{batch}.net")

                print(f"epoch: {epoch}, batch: {batch}, loss: {total_loss/args.frequency:.3f}")
                print(f"took {time.time() - start} seconds")
                batches, total_loss, start = 0, 0, time.time()

