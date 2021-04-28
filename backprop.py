import argparse, pickle, time, sys
import torch
import torch.nn as nn
from torchvision import transforms

"""
backpropagation on an image for image generation
trying to iteratively sample a random image from the
implicit image manifold parameterized by the discriminator
- initialize random image, there are a variety of setups:
normal distribution, uniform on [-1, 1], etc.
- run through discriminator to get a probability
- optimize image by backprop on the objective -log(D(X))
- moves image to high probability image
"""

torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backpropagation for image generation.")
    parser.add_argument("-v", "--version", action="version", version="back 1.0")
    parser.add_argument("-n", "--network", help="path to saved network")
    parser.add_argument("-e", "--epochs", type=int, default=10**3)
    parser.add_argument("-f", "--frequency", type=int, default=10**2,
                        help="number of batches until printing a summary")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    # generate random image
    im = torch.rand((3, 256, 256), requires_grad=False, device="cuda")
    im = 2*im - 1
    im = im.requires_grad_(True)

    # load models
    with open(args.network, "rb") as f:
        data = pickle.load(f)
        G, D = data["G_ema"].cuda(), data["D"].cuda()

    optimizer = torch.optim.Adam([im], lr=args.learning_rate)
    batches, total_loss, total_p, start = 0, 0, 0, time.time()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # forward
        logit = D(im.reshape(1, *im.shape), None)
        p = torch.sigmoid(logit)
        loss = torch.nn.functional.softplus(-logit)

        total_loss, total_p = total_loss + loss.item(), total_p + p.item()
        batches += 1

        # backward
        loss.backward()
        optimizer.step()

        # summary
        if batches == args.frequency:
            # convert to image
            img = (127.5*im + 128).clamp(0, 255).to(torch.uint8)
            img = transforms.functional.to_pil_image(img)
            img.save(f"train/img{epoch:04d}.png")

            print(f"epoch: {epoch:04d}, loss: {total_loss/args.frequency:.3f}, p: {total_p/args.frequency:.3f}")
            print(f"took {time.time() - start} seconds")
            sys.stdout.flush()
            batches, total_loss, total_p, start = 0, 0, 0, time.time()

