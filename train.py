import argparse
import os
import torch
from torch import nn, optim
from generator import Generator
from discriminator import Discriminator
from dataset import get_cifar10_dataloader
import matplotlib.pyplot as plt
from torchvision import transforms


def train(dataset_path, epochs=50, batch_size=16, resize=128, save_path="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Завантаження моделей
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Оптимізатори
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

    # Функції втрат
    adversarial_loss = nn.BCELoss()
    reconstruction_loss = nn.MSELoss()

    # Підготовка даних із трансформацією для зміни розміру
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),  # Зміна розміру зображення
        transforms.ToTensor()
    ])
    dataloader, _ = get_cifar10_dataloader(
        batch_size=batch_size, transform=transform)

    # Логування втрат
    gen_losses = []
    disc_losses = []

    # Створення директорії для збереження
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        epoch_gen_loss = 0.0
        epoch_disc_loss = 0.0

        for batch_idx, (low_res, high_res) in enumerate(dataloader, start=1):
            # Для простоти high_res дорівнює low_res
            low_res, high_res = low_res.to(
                device).float(), low_res.to(device).float()

            # Логування форми даних
            print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}: "
                  f"low_res shape: {low_res.shape}, high_res shape: {high_res.shape}")

            # Навчання дискримінатора
            real_labels = torch.ones((low_res.size(0), 1), device=device)
            fake_labels = torch.zeros((low_res.size(0), 1), device=device)

            fake_high_res = generator(low_res)

            # Передача в дискримінатор
            real_loss = adversarial_loss(discriminator(high_res), real_labels)
            fake_loss = adversarial_loss(discriminator(
                fake_high_res.detach()), fake_labels)
            disc_loss = (real_loss + fake_loss) / 2

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Навчання генератора
            gen_loss = reconstruction_loss(fake_high_res, high_res) + \
                adversarial_loss(discriminator(fake_high_res), real_labels)

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()

            # Прогрес батчів
            if batch_idx % 10 == 0 or batch_idx == len(dataloader):
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{len(dataloader)}: "
                      f"Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}")

        # Логування втрат за епоху
        gen_losses.append(epoch_gen_loss / len(dataloader))
        disc_losses.append(epoch_disc_loss / len(dataloader))

        print(f"Epoch [{epoch + 1}/{epochs}] complete. "
              f"Avg Generator Loss: {gen_losses[-1]:.4f}, Avg Discriminator Loss: {disc_losses[-1]:.4f}")

        # Збереження моделей після кожної епохи
        torch.save(generator.state_dict(), os.path.join(
            save_path, f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(
            save_path, f"discriminator_epoch_{epoch+1}.pth"))

    # Побудова графіків втрат
    plt.figure()
    plt.plot(range(1, epochs + 1), gen_losses, label="Generator Loss")
    plt.plot(range(1, epochs + 1), disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("training_losses.png")
    print("Графік втрат збережено як training_losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GAN for image restoration.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--resize", type=int, default=128,
                        help="Resize images to this size.")
    parser.add_argument("--save_path", type=str,
                        default="checkpoints", help="Path to save the checkpoints.")
    args = parser.parse_args()

    train(dataset_path=args.dataset_path, epochs=args.epochs,
          batch_size=args.batch_size, resize=args.resize, save_path=args.save_path)
