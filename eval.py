import torch
from torch.utils.data import TensorDataset
from models import AsymVLMPSD, AsymVLMVMF, ProbVLM, PFE, PCMEPP, ProLIP
import argparse
import json
import os


DEVICE = "cuda:0"


def t2i_acc(img_dataset, text_dataset, cap_id_to_img_id, adaptor, num_uncer_levels):
    # get predictions
    log_likelihood = compute_likelihood(img_dataset, text_dataset, adaptor)
    all_predictions = log_likelihood.argmax(dim=0)

    # build the targets
    all_targets = []
    for cap_id in range(len(text_dataset)):
        img_id = cap_id_to_img_id[str(cap_id)]
        all_targets.append(img_id)

    # compute the text uncertainty
    uncertainties = adaptor.adapt_text(text_dataset.tensors[0].float().to(DEVICE))[1]
    uncertainties = uncertainties.cpu().numpy()

    # sort the predictions based on the uncertainty
    # 1. sort the uncertainties
    sorted_indices = uncertainties.argsort()
    # 2. group the uncertainties
    group_size = len(uncertainties) // num_uncer_levels

    # 3. compute the accuracy for each group
    accs = []
    for i in range(num_uncer_levels):
        indices = sorted_indices[i * group_size: (i + 1) * group_size]
        predictions = all_predictions[indices]
        targets = [all_targets[idx] for idx in indices]

        # compute the accuracy
        targets = torch.tensor(targets).to(DEVICE)
        correct = (predictions == targets).sum().item()

        accs.append(correct / len(targets))

    return accs


def i2t_acc(img_dataset, text_dataset, cap_id_to_img_id, adaptor, num_uncer_levels=5):
    # get the predictions
    log_likelihood = compute_likelihood(img_dataset, text_dataset, adaptor)
    all_predictions = log_likelihood.argmax(dim=1)

    # build the targets
    all_targets = {
        img_id: []
        for img_id in range(len(img_dataset))
    }
    for cap_id in range(len(text_dataset)):
        img_id = cap_id_to_img_id[str(cap_id)]
        all_targets[img_id].append(cap_id)

    # compute the text uncertainty
    uncertainties = adaptor.adapt_text(text_dataset.tensors[0].float().to(DEVICE))[1]
    uncertainties = uncertainties.cpu().numpy()[all_predictions.cpu().numpy()]

    # sort the predictions based on the uncertainty
    # 1. sort the uncertainties
    sorted_indices = uncertainties.argsort()
    # 2. group the uncertainties
    group_size = len(uncertainties) // num_uncer_levels

    # 3. compute the accuracy for each group
    accs = []
    for i in range(num_uncer_levels):
        indices = sorted_indices[i * group_size: (i + 1) * group_size]
        predictions = all_predictions[indices]
        targets = [all_targets[idx] for idx in indices]

        # compute the accuracy
        correct = 0
        for img_id, prediction in enumerate(predictions):
            correct += (prediction in targets[img_id])

        accs.append(correct / len(targets))

    return accs


def compute_likelihood(img_dataset, text_dataset, adaptor):
    img_embeddings = img_dataset.tensors[0].float().to(DEVICE)
    text_embedding = text_dataset.tensors[0].float().to(DEVICE)

    log_likelihood = adaptor.log_likelihood(img_embeddings, text_embedding)

    return log_likelihood


@torch.no_grad()
def main(args):
    adaptor_name = args.method
    dataset = args.dataset
    seed = args.seed

    Adaptor = {
        'asymvlm-psd': AsymVLMPSD,
        'asymvlm-vmf': AsymVLMVMF,
        'probvlm': ProbVLM,
        'pfe': PFE,
        'pcmepp': PCMEPP,
        'prolip': ProLIP,
    }

    device = "cuda:0"
    adaptor = Adaptor[adaptor_name]().to(device)

    adaptor.load_state_dict(
        torch.load(f"checkpoints/{dataset}/{seed}/{adaptor_name}.pt", map_location=DEVICE))
    adaptor.eval()

    text_dataset = TensorDataset(torch.load(f'embeddings/{dataset}/val/text.pt'))
    img_dataset = TensorDataset(torch.load(f'embeddings/{dataset}/val/image.pt'))

    with open(f'embeddings/{dataset}/val/cap_id_to_img_id.json', 'r') as f:
        cap_id_to_img_id = json.load(f)

    t2i_accs = t2i_acc(img_dataset, text_dataset, cap_id_to_img_id, adaptor, args.uncer_levels)
    i2t_accs = i2t_acc(img_dataset, text_dataset, cap_id_to_img_id, adaptor, args.uncer_levels)

    # save the results
    result_dir = f"results/{dataset}/{seed}"
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/{adaptor_name}.json", "w") as f:
        json.dump({
            "t2i": t2i_accs,
            "i2t": i2t_accs
        }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument(
        '--method', type=str, default='asymvlm-psd',)
    parser.add_argument(
        '--seed', type=int, default=0,
        help='random seed for the training of the adaptor')

    parser.add_argument("--uncer_levels", type=int, default=10)
    args = parser.parse_args()

    main(args)