import os
import torch
from torch.nn.functional import interpolate
from generator import Generator
from dataset import get_cifar10_dataloader
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np


def save_comparison(low_res, high_res, restored, save_path, epoch, index):
    """Зберігає порівняльне зображення (low_res, high_res, restored)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    low_res = np.transpose(low_res.numpy(), (1, 2, 0))
    high_res = np.transpose(high_res.numpy(), (1, 2, 0))
    restored = np.transpose(restored.numpy(), (1, 2, 0))

    axes[0].imshow(np.clip(low_res, 0, 1))
    axes[0].set_title("Low Resolution")
    axes[0].axis("off")

    axes[1].imshow(np.clip(high_res, 0, 1))
    axes[1].set_title("Original (High Res)")
    axes[1].axis("off")

    axes[2].imshow(np.clip(restored, 0, 1))
    axes[2].set_title("Restored")
    axes[2].axis("off")

    plt.tight_layout()
    filename = f"{save_path}/comparison_epoch_{epoch}_image_{index}.png"
    plt.savefig(filename)
    plt.close()


def evaluate(generator_model_path, dataset_path, batch_size=16, max_images=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(generator_model_path):
        print(f"Файл моделі {
              generator_model_path} не знайдено. Пропускаємо оцінку.")
        return

    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()

    dataloader, _ = get_cifar10_dataloader(batch_size=batch_size)

    total_psnr, total_ssim = 0, 0
    count = 0

    os.makedirs("comparisons", exist_ok=True)
    epoch = int(generator_model_path.split("_epoch_")[1].split(".pth")[0])
    saved_images = 0

    for batch_idx, (low_res, _) in enumerate(dataloader):
        if saved_images >= max_images:
            break  # Зупиняємося після збереження 5 зображень

        low_res = low_res.to(device)

        with torch.no_grad():
            restored = generator(low_res).cpu()

        high_res = low_res.cpu()  # Використовуємо оригінал як high_res

        for i in range(min(max_images - saved_images, low_res.size(0))):
            save_comparison(
                low_res[i].cpu(),
                high_res[i].cpu(),
                restored[i].cpu(),
                save_path="comparisons",
                epoch=epoch,
                index=saved_images
            )
            saved_images += 1

        total_psnr += psnr(high_res.numpy(), restored.numpy(), data_range=1.0)
        total_ssim += ssim(
            high_res.numpy().transpose(0, 2, 3, 1),
            restored.numpy().transpose(0, 2, 3, 1),
            win_size=3,
            channel_axis=-1,
            data_range=1.0
        )
        count += 1

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        result = f"Average PSNR: {
            avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}\n"
        print(result)

        with open("evaluation_results.txt", "a") as f:
            f.write(f"Model: {generator_model_path}\n")
            f.write(result)
    else:
        print("Оцінка не виконана: тестовий датасет порожній.")


def evaluate_multiple_models(model_dir, dataset_path, batch_size=16, max_images=5):
    model_files = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("generator_epoch_") and f.endswith(".pth")
    ])

    if not model_files:
        print("У каталозі немає файлів моделей генератора.")
        return

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        print(f"\nОцінка моделі: {model_path}")
        evaluate(generator_model_path=model_path, dataset_path=dataset_path,
                 batch_size=batch_size, max_images=max_images)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate GAN model performance.")
    parser.add_argument("--gan_model_path", type=str,
                        help="Path to a single GAN model.")
    parser.add_argument("--model_dir", type=str,
                        help="Path to a directory with GAN models.")
    parser.add_argument("--dataset_path", type=str,
                        required=True, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_images", type=int, default=5,
                        help="Maximum number of comparison images per model.")
    args = parser.parse_args()

    if args.gan_model_path:
        evaluate(generator_model_path=args.gan_model_path, dataset_path=args.dataset_path,
                 batch_size=args.batch_size, max_images=args.max_images)
    elif args.model_dir:
        evaluate_multiple_models(model_dir=args.model_dir, dataset_path=args.dataset_path,
                                 batch_size=args.batch_size, max_images=args.max_images)
    else:
        print("Необхідно вказати або `--gan_model_path`, або `--model_dir`.")
