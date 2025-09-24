import torch
import clip
from datasets.embedding import EmbeddingDataset
from models import ProbVLM, PFE, AsymVLMVMF, AsymVLMPSD, PCMEPP, ProLIP
from torch.utils.data import DataLoader
import os
import argparse
from utils.seed import set_seed


def main(args):
    data = args.dataset
    method = args.method
    seed = args.seed

    set_seed(seed)
    Adaptor = {
        'asymvlm-psd': AsymVLMPSD,
        'asymvlm-vmf': AsymVLMVMF,
        'probvlm': ProbVLM,
        'pfe': PFE,
        'pcmepp': PCMEPP,
        'prolip': ProLIP,
    }

    device = "cuda:0"
    model, _ = clip.load("ViT-B/32", device=device)
    adaptor = Adaptor[method]().to(device)

    dataset = EmbeddingDataset(
        f'embeddings/{data}/train/image.pt',
        f'embeddings/{data}/train/text.pt',
        f'embeddings/{data}/train/cap_id_to_img_id.json')

    num_workers = 16
    batch_size = 2048

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    num_epochs = 200

    if 'asymvlm' in method:
        optimizer = torch.optim.SGD(adaptor.parameters(),
            lr=1e-2, momentum=0.9, weight_decay=5e-4)
    elif method == 'probvlm':
        optimizer = torch.optim.AdamW(adaptor.parameters(),
            lr=1e-4, weight_decay=5e-4)
    elif method == 'pfe':
        optimizer = torch.optim.SGD(adaptor.parameters(),
            lr=1e-3, momentum=0.9, weight_decay=5e-4)
    elif method == 'pcmepp':
        optimizer = torch.optim.AdamW(adaptor.parameters(),
            lr=5e-4, weight_decay=1e-4)
    elif method == 'prolip':
        optimizer = torch.optim.AdamW(adaptor.parameters(),
            lr=5e-4, weight_decay=5e-4)
    else:
        raise ValueError(f'Unknown method: {method}')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs*len(dataloader), eta_min=1e-6)


    for epoch in range(num_epochs):
        adaptor.train()

        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)

            output = adaptor(images, captions)
            loss = adaptor.loss(*output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # save checkpoint
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                adaptor.eval()
                sentences = [
                    "a photo",
                    "a photo of a cat",
                    "a photo of a black cat",
                    "a photo of a black cat with green eyes",
                    "a photo of a black cat with big dark green eyes ",
                ]
                text = clip.tokenize(sentences).to(device)
                _, uncertainties = adaptor.adapt_text(model.encode_text(text).float())
                # print(f'Epoch {epoch+1} with tempperature {adaptor.text_adaptor.temperature.item():.4f}')
                for j in range(len(sentences)):
                    print(f'{sentences[j]}: {uncertainties[j].item():.4e}')
                print("============", flush=True)

            save_dir = f'checkpoints/{data}/{seed}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(adaptor.state_dict(), f'{save_dir}/{method}.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--method', type=str, default='asymvlm-psd',)
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed for initialization')
    args = parser.parse_args()

    main(args)