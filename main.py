import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

manualSeed = 999  # Set random seed for reproducibility
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Parameters
dataroot = "data/celeba"  # Root directory for dataset
workers = 1  # Number of workers for dataloader
batch_size = 16  # Batch size during training
image_size = 64  # Spatial size of training images. All images will be resized to this size using a transformer.
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 5  # Number of training epochs
lr = 0.0002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
sample_interval = 1  # Number of epochs before we sample some images

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Create the dataset. We can use an image folder dataset the way we have it setup.
def setupDataloader(dataroot, image_size, batch_size, workers):
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    return dataloader


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Reusable Convolutional Block
        def convBlock(inc, outc, ksz, stride, dilation, bias):
            return nn.Sequential(
                nn.ConvTranspose2d(inc, outc, ksz, stride, dilation, bias=bias),
                nn.BatchNorm2d(outc),
                nn.ReLU(True)
            )

        # Generator Model
        self.main = nn.Sequential(
            convBlock(nz, ngf * 8, 4, 1, 0, False),  # input is Z, going into a convolution

            convBlock(ngf * 8, ngf * 4, 4, 2, 1, False),  # state size. (ngf*8) x 4 x 4
            convBlock(ngf * 4, ngf * 2, 4, 2, 1, False),  # state size. (ngf*4) x 8 x 8
            convBlock(ngf * 2, ngf * 1, 4, 2, 1, False),  # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # state size. (ngf) x 32 x 32
            nn.Tanh()  # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Reusable convolutional block
        def convBlock(inc, outc, ksz, stride, dilation, bias):
            return nn.Sequential(
                nn.Conv2d(inc, outc, ksz, stride, dilation, bias=bias),
                nn.BatchNorm2d(outc),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            convBlock(ndf, ndf * 2, 4, 2, 1, False),  # state size. (ndf) x 32 x 32
            convBlock(ndf * 2, ndf * 4, 4, 2, 1, False),  # state size. (ndf*2) x 16 x 16
            convBlock(ndf * 4, ndf * 8, 4, 2, 1, False),  # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # state size. (ndf*8) x 4 x 4
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # Create Generator and Discriminator
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(generator, list(range(ngpu)))
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(discriminator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Print the model
    print(generator)
    print(discriminator)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    dataloader = setupDataloader(dataroot, image_size, batch_size, workers)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()  # zero grad discriminator optimizer

            # 1 - Train on a batch of real images
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = discriminator(real_cpu).view(-1)  # Forward pass real batch through D
            errD_real = criterion(output, label)  # Calculate loss on all-real batch
            errD_real.backward()  # Calculate gradients for D in backward pass
            D_x = output.mean().item()

            # 2 - Train on a batch of fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)  # Generate batch of latent vectors (noise input)

            fake = generator(noise)  # Generate fake image batch with G
            label.fill_(fake_label)  # Update labels to indicate fake images
            output = discriminator(fake.detach()).view(-1)  # Classify all fake batch with D
            errD_fake = criterion(output, label)  # Calculate D's loss on the all-fake batch
            errD_fake.backward()  # Calculate the gradients for this batch
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake  # Add the gradients from the all-real and all-fake batches
            optimizerD.step()  # Update D

            # (2) Update G network: maximize log(D(G(z)))
            generator.zero_grad()  # zero grad generator optimizer
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)  # Calculate G's loss based on this output
            errG.backward()  # Calculate gradients for G
            D_G_z2 = output.mean().item()

            optimizerG.step()  # Update G

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % sample_interval == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                plt.imsave('imgs/faces_epoch{}_batch{}.jpg'.format(epoch, i),
                           np.transpose(img_list[-1], (1, 2, 0)).numpy())
                plt.close('all')

            iters += 1
